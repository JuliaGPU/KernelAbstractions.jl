# Volume integral of a discontinuous Galerkin spectral element method (DGSEM), as in
# Trixi.jl: quadrilateral (2D) or hexahedral (3D) elements with `N` nodes per direction,
# the compressible Euler equations (`NV = DIM + 2` variables) and curvilinear metric
# terms, with the solution stored as `u[v, i, j, (k,) element]`. For the weak form, each
# node computes the flux of its state in the direction of each of its contravariant
# vectors `Ja[:, d, i, j, (k,) element]` (one division per node), and
#
#     du[:, i, j, k, e] = Σₗ D[i, l] F¹[:, l, j, k, e] + Σₗ D[j, l] F²[:, i, l, k, e]
#                       + Σₗ D[k, l] F³[:, i, j, l, e]
#
# The work per element is small and the data of an element (and its fluxes) has to be
# shared between all of its nodes, so there are several ways to map it to work-items. The
# variants below compute the same `du` (up to rounding) and measure:
#
# - "weak/element": one work-item per element, which loops over the nodes and scatters
#   their contributions into `du` in global memory, i.e., read-modify-write. This is the
#   CPU code run per element, which is what Trixi's generic GPU fallback does.
# - "weak/element private": the same, but accumulated in an `@private` array of the size
#   of an element, which is written to `du` once. Measures `@private` memory that is too
#   large for registers.
# - "weak/node localmem E=<E>": one work-item per node, workgroups of `E` elements. Each
#   work-item stores its fluxes in `@localmem`, and after a `@synchronize` sums along its
#   lines and writes its `du` once. With `E = 1` the workgroups have only `N^DIM`
#   work-items; the larger `E` has 128 per workgroup, as in Trixi's GPU kernels.
# - "weak/node recompute E=<E>": one work-item per node, but without local memory or
#   `@synchronize`: each work-item recomputes the fluxes of the nodes on its lines (`DIM *
#   N` instead of `DIM` fluxes). The control for the cost of `@synchronize`.
# - "weak/slab E=<E>" (3D): a workgroup of `N × N` work-items per element (as in
#   CUDA.jl's `perf/volumerhs.jl` and Raven.jl), which loops over the `k` slabs: the fluxes
#   in the `i` and `j` directions go through `O(N²)` local memory per element, while the
#   contributions in the `k` direction and the result are accumulated in `@private`
#   arrays. Two `@synchronize` per slab.
# - "weak/slab double-buffered E=<E>" (3D): the same with two local-memory buffers that
#   alternate between slabs, which needs only one `@synchronize` per slab, as in the
#   slab kernel prototyped for Trixi.jl's 3D `P4estMesh` (trixi-framework/Trixi.jl#3339).
# - "fluxdiff/element" and "fluxdiff/node E=<E>": the flux differencing form, with a symmetric
#   two-point flux of the averaged state of two nodes on a line instead of the
#   derivative of the flux. Per element, each two-point flux is computed once and added
#   to both nodes (read-modify-write in global memory); per node, each work-item computes
#   all the two-point fluxes of its node, i.e., each one twice, and writes `du` once.
# - "fluxdiff/node halfsweep E=<E>": one work-item per node, but each two-point flux is
#   computed once, by the first node of the pair, and shared with the second through
#   `@localmem` (`O(N^(DIM + 1))` per element) and a `@synchronize`, as Trixi's `HalfSweep`
#   kernels.
#
# Local-memory layout: the kernels with local memory store the variables of a node either
# first, `F[v, i, j, (k,) e, ...]`, or last, `F[i, j, (k,) e, ..., v]`. With the variable
# index last, neighbouring work-items access neighbouring words, while with it first and an
# even number of variables (`NV = 4` in 2D) they access the same shared-memory banks on
# NVIDIA GPUs (trixi-framework/Trixi.jl#3337). On the CPU the variables of a node in
# consecutive words are faster. So, like the `VIJFH` and `VIJHF` data layouts of
# ClimaCore.jl and a prototype for Trixi.jl, each backend gets a default layout
# (`dgsem_default_layout`): the variable index first on the CPU and last otherwise. The variants without a layout in
# their name use the default, and those named "v-first" or "v-last" the other layout.
#
# With a KernelAbstractions that has tiles (`@kernel tile`), `dgsem_tiles.jl` adds variants
# that exchange values between the work-items of an element with `tile_shfl` instead.
#
# These are compute-heavy kernels with small, `@synchronize`d workgroups and `@private`
# arrays, so they are sensitive to how a back-end handles local memory, barriers and
# private memory, unlike the bandwidth-bound kernels of the other groups. The values of `D` and `Ja`
# don't matter for the timing, so they are random.

using KernelAbstractions.Extras: @unroll

# The state of a node, `A[1:NV, I...]`, as a tuple
@inline dgsem_node(A, ::Val{NV}, I::Vararg{Integer}) where {NV} =
    ntuple(v -> @inbounds(A[v, I...]), Val(NV))

# The contravariant vector `d` of a node, `Ja[1:DIM, d, I...]`
@inline dgsem_metric(Ja, ::Val{DIM}, d, I::Vararg{Integer}) where {DIM} =
    ntuple(dd -> @inbounds(Ja[dd, d, I...]), Val(DIM))

# Flux of the Euler equations (γ = 1.4) in the direction `n`, i.e., `Σ_d n[d] f_d(u)`
@inline function dgsem_flux(u::NTuple{NV, T}, n::NTuple{DIM, T}) where {NV, DIM, T}
    ρ, E = u[1], u[NV]
    m = ntuple(d -> u[1 + d], Val(DIM))
    ρinv = inv(ρ)
    p = T(0.4) * (E - T(0.5) * ρinv * sum(m .* m))
    mn = sum(m .* n)
    vn = mn * ρinv
    return (mn, ntuple(d -> muladd(m[d], vn, p * n[d]), Val(DIM))..., (E + p) * vn)
end

# A symmetric two-point flux (Kennedy and Gruber) in the direction `n`
@inline function dgsem_twopoint_flux(
        uL::NTuple{NV, T}, uR::NTuple{NV, T}, n::NTuple{DIM, T}
    ) where {NV, DIM, T}
    ρinvL, ρinvR = inv(uL[1]), inv(uR[1])
    vL = ntuple(d -> uL[1 + d] * ρinvL, Val(DIM))
    vR = ntuple(d -> uR[1 + d] * ρinvR, Val(DIM))
    pL = T(0.4) * (uL[NV] - T(0.5) * uL[1] * sum(vL .* vL))
    pR = T(0.4) * (uR[NV] - T(0.5) * uR[1] * sum(vR .* vR))
    ρ = T(0.5) * (uL[1] + uR[1])
    v = T(0.5) .* (vL .+ vR)
    p = T(0.5) * (pL + pR)
    e = T(0.5) * (uL[NV] * ρinvL + uR[NV] * ρinvR)
    vn = sum(v .* n)
    f1 = ρ * vn
    return (f1, ntuple(d -> muladd(f1, v[d], p * n[d]), Val(DIM))..., muladd(f1, e, p * vn))
end

@inline dgsem_average(a, b) = (a .+ b) .* convert(eltype(a), 0.5)

@inline dgsem_zero(::Type{T}, ::Val{NV}) where {T, NV} = ntuple(_ -> zero(T), Val(NV))

# The nodes of an element
@inline dgsem_nodes(::Val{N}, ::Val{DIM}) where {N, DIM} =
    CartesianIndices(ntuple(_ -> N, Val(DIM)))

# The node `l` of the line through `node` in direction `d`
@inline dgsem_line(node, d, l) = Base.setindex(node, l, d)

# The layout of the variables in local memory: `F[v, I...]` or `F[I..., v]`
abstract type DGSEMLayout end
struct DGSEMVariableFirst <: DGSEMLayout end
struct DGSEMVariableLast <: DGSEMLayout end

# The layout that a backend prefers (see "Local-memory layout" above)
dgsem_default_layout(backend) = backend isa CPU ? DGSEMVariableFirst() : DGSEMVariableLast()

# The suffix of the name of a variant with `layout` on `backend`: none for the default
dgsem_layout_name(backend, layout) =
    layout == dgsem_default_layout(backend) ? "" :
    layout isa DGSEMVariableLast ? " v-last" : " v-first"

# The index of variable `v` at `I` in local memory
@inline dgsem_local_index(::DGSEMVariableFirst, v, I...) = (v, I...)
@inline dgsem_local_index(::DGSEMVariableLast, v, I...) = (I..., v)

# The state of a node in local memory, as a tuple
@inline dgsem_local_node(A, layout::DGSEMLayout, ::Val{NV}, I::Vararg{Integer}) where {NV} =
    ntuple(v -> @inbounds(A[dgsem_local_index(layout, v, I...)...]), Val(NV))

@kernel inbounds = true function dgsem_weak_element!(
        du, @Const(u), @Const(D), @Const(Ja), ::Val{N}, ::Val{DIM}
    ) where {N, DIM}
    NV = DIM + 2
    e = @index(Global)
    for I in dgsem_nodes(Val(N), Val(DIM))
        @unroll for v in 1:NV
            du[v, Tuple(I)..., e] = zero(eltype(du))
        end
    end
    for I in dgsem_nodes(Val(N), Val(DIM))
        node = Tuple(I)
        U = dgsem_node(u, Val(NV), node..., e)
        @unroll for d in 1:DIM
            F = dgsem_flux(U, dgsem_metric(Ja, Val(DIM), d, node..., e))
            for l in 1:N
                Dl = D[l, node[d]]
                target = dgsem_line(node, d, l)
                @unroll for v in 1:NV
                    du[v, target..., e] = muladd(Dl, F[v], du[v, target..., e])
                end
            end
        end
    end
end

@kernel inbounds = true function dgsem_weak_element_private!(
        du, @Const(u), @Const(D), @Const(Ja), ::Val{N}, ::Val{DIM}
    ) where {N, DIM}
    @uniform NV = DIM + 2
    e = @index(Global)
    acc = @private eltype(du) (NV, ntuple(_ -> N, Val(DIM))...)
    for I in dgsem_nodes(Val(N), Val(DIM))
        @unroll for v in 1:NV
            acc[v, Tuple(I)...] = zero(eltype(du))
        end
    end
    for I in dgsem_nodes(Val(N), Val(DIM))
        node = Tuple(I)
        U = dgsem_node(u, Val(NV), node..., e)
        @unroll for d in 1:DIM
            F = dgsem_flux(U, dgsem_metric(Ja, Val(DIM), d, node..., e))
            for l in 1:N
                Dl = D[l, node[d]]
                target = dgsem_line(node, d, l)
                @unroll for v in 1:NV
                    acc[v, target...] = muladd(Dl, F[v], acc[v, target...])
                end
            end
        end
    end
    for I in dgsem_nodes(Val(N), Val(DIM))
        @unroll for v in 1:NV
            du[v, Tuple(I)..., e] = acc[v, Tuple(I)...]
        end
    end
end

# Workgroups of `(N, ..., N, E)` work-items, i.e., `E` elements
@kernel inbounds = true function dgsem_weak_node_localmem!(
        du, @Const(u), @Const(D), @Const(Ja), ::Val{N}, ::Val{DIM}, ::Val{E}, layout::DGSEMLayout
    ) where {N, DIM, E}
    @uniform NV = DIM + 2
    I = @index(Global, NTuple)
    node, e = Base.front(I), last(I)
    le = last(@index(Local, NTuple))
    F = @localmem eltype(du) dgsem_local_index(layout, NV, ntuple(_ -> N, Val(DIM))..., E, DIM)
    U = dgsem_node(u, Val(NV), node..., e)
    @unroll for d in 1:DIM
        G = dgsem_flux(U, dgsem_metric(Ja, Val(DIM), d, node..., e))
        @unroll for v in 1:NV
            F[dgsem_local_index(layout, v, node..., le, d)...] = G[v]
        end
    end
    @synchronize
    acc = dgsem_zero(eltype(du), Val(NV))
    for l in 1:N
        @unroll for d in 1:DIM
            G = dgsem_local_node(F, layout, Val(NV), dgsem_line(node, d, l)..., le, d)
            acc = muladd.(D[node[d], l], G, acc)
        end
    end
    @unroll for v in 1:NV
        du[v, node..., e] = acc[v]
    end
end

@kernel inbounds = true function dgsem_weak_node_recompute!(
        du, @Const(u), @Const(D), @Const(Ja), ::Val{N}, ::Val{DIM}
    ) where {N, DIM}
    NV = DIM + 2
    I = @index(Global, NTuple)
    node, e = Base.front(I), last(I)
    acc = dgsem_zero(eltype(du), Val(NV))
    for l in 1:N
        @unroll for d in 1:DIM
            source = dgsem_line(node, d, l)
            G = dgsem_flux(
                dgsem_node(u, Val(NV), source..., e),
                dgsem_metric(Ja, Val(DIM), d, source..., e)
            )
            acc = muladd.(D[node[d], l], G, acc)
        end
    end
    @unroll for v in 1:NV
        du[v, node..., e] = acc[v]
    end
end

# The work-item `(i, j)` of an element owns the column `du[:, i, j, :, e]`, which it
# accumulates in `@private` memory over the slabs `k`: the flux in the `k` direction of
# node `(i, j, k)` only contributes to its own column, the fluxes in the `i` and `j`
# directions are shared through local memory, one slab at a time. The loops over `k` and
# `l` are unrolled so that the private arrays can be kept in registers. With `NB = 1`
# local buffer, a slab waits until all work-items are done reading the previous one, i.e.,
# two `@synchronize` per slab. With `NB = 2`, consecutive slabs use different buffers and
# the `@synchronize` of slab `k + 1` ensures that slab `k` has been read before slab `k + 2`
# overwrites its buffer, so one `@synchronize` per slab suffices.
@kernel inbounds = true function dgsem_weak_slab_3d!(
        du, @Const(u), @Const(D), @Const(Ja), ::Val{N}, ::Val{E}, ::Val{NB}, layout::DGSEMLayout
    ) where {N, E, NB}
    @uniform NV = 5
    i, j, e = @index(Global, NTuple)
    _, _, le = @index(Local, NTuple)
    F1 = @localmem eltype(du) dgsem_local_index(layout, NV, N, N, E, NB)
    F2 = @localmem eltype(du) dgsem_local_index(layout, NV, N, N, E, NB)
    acc = @private eltype(du) (NV, N)
    @unroll for k in 1:N
        @unroll for v in 1:NV
            acc[v, k] = zero(eltype(du))
        end
    end
    @unroll for k in 1:N
        b = mod1(k, NB)
        U = dgsem_node(u, Val(NV), i, j, k, e)
        G1 = dgsem_flux(U, dgsem_metric(Ja, Val(3), 1, i, j, k, e))
        G2 = dgsem_flux(U, dgsem_metric(Ja, Val(3), 2, i, j, k, e))
        G3 = dgsem_flux(U, dgsem_metric(Ja, Val(3), 3, i, j, k, e))
        if NB == 1
            # all work-items are done reading the previous slab
            @synchronize
        end
        @unroll for v in 1:NV
            F1[dgsem_local_index(layout, v, i, j, le, b)...] = G1[v]
            F2[dgsem_local_index(layout, v, i, j, le, b)...] = G2[v]
        end
        @unroll for l in 1:N
            Dlk = D[l, k]
            @unroll for v in 1:NV
                acc[v, l] = muladd(Dlk, G3[v], acc[v, l])
            end
        end
        @synchronize
        @unroll for l in 1:N
            Dil, Djl = D[i, l], D[j, l]
            @unroll for v in 1:NV
                acc[v, k] = muladd(Dil, F1[dgsem_local_index(layout, v, l, j, le, b)...], acc[v, k])
                acc[v, k] = muladd(Djl, F2[dgsem_local_index(layout, v, i, l, le, b)...], acc[v, k])
            end
        end
    end
    @unroll for k in 1:N
        @unroll for v in 1:NV
            du[v, i, j, k, e] = acc[v, k]
        end
    end
end

# `Ds` has a zero diagonal, as Trixi's `derivative_split`
@kernel inbounds = true function dgsem_fluxdiff_element!(
        du, @Const(u), @Const(Ds), @Const(Ja), ::Val{N}, ::Val{DIM}
    ) where {N, DIM}
    NV = DIM + 2
    e = @index(Global)
    for I in dgsem_nodes(Val(N), Val(DIM))
        @unroll for v in 1:NV
            du[v, Tuple(I)..., e] = zero(eltype(du))
        end
    end
    for I in dgsem_nodes(Val(N), Val(DIM))
        node = Tuple(I)
        U = dgsem_node(u, Val(NV), node..., e)
        @unroll for d in 1:DIM
            n = dgsem_metric(Ja, Val(DIM), d, node..., e)
            for l in (node[d] + 1):N
                other = dgsem_line(node, d, l)
                f = dgsem_twopoint_flux(
                    U, dgsem_node(u, Val(NV), other..., e),
                    dgsem_average(n, dgsem_metric(Ja, Val(DIM), d, other..., e))
                )
                Dnl, Dln = Ds[node[d], l], Ds[l, node[d]]
                @unroll for v in 1:NV
                    du[v, node..., e] = muladd(Dnl, f[v], du[v, node..., e])
                    du[v, other..., e] = muladd(Dln, f[v], du[v, other..., e])
                end
            end
        end
    end
end

@kernel inbounds = true function dgsem_fluxdiff_node!(
        du, @Const(u), @Const(Ds), @Const(Ja), ::Val{N}, ::Val{DIM}
    ) where {N, DIM}
    NV = DIM + 2
    I = @index(Global, NTuple)
    node, e = Base.front(I), last(I)
    U = dgsem_node(u, Val(NV), node..., e)
    acc = dgsem_zero(eltype(du), Val(NV))
    @unroll for d in 1:DIM
        n = dgsem_metric(Ja, Val(DIM), d, node..., e)
        for l in 1:N
            if l != node[d]
                other = dgsem_line(node, d, l)
                f = dgsem_twopoint_flux(
                    U, dgsem_node(u, Val(NV), other..., e),
                    dgsem_average(n, dgsem_metric(Ja, Val(DIM), d, other..., e))
                )
                acc = muladd.(Ds[node[d], l], f, acc)
            end
        end
    end
    @unroll for v in 1:NV
        du[v, node..., e] = acc[v]
    end
end

# Each work-item computes the two-point fluxes with the nodes after it on its lines and
# stores them in local memory, at `F[v, node..., e, l, d]` for the node `l` in direction
# `d`. After a `@synchronize`, it reads the fluxes with the nodes before it from theirs.
# Workgroups of `(N, ..., N, E)` work-items.
@kernel inbounds = true function dgsem_fluxdiff_halfsweep!(
        du, @Const(u), @Const(Ds), @Const(Ja), ::Val{N}, ::Val{DIM}, ::Val{E}, layout::DGSEMLayout
    ) where {N, DIM, E}
    @uniform NV = DIM + 2
    I = @index(Global, NTuple)
    node, e = Base.front(I), last(I)
    le = last(@index(Local, NTuple))
    F = @localmem eltype(du) dgsem_local_index(layout, NV, ntuple(_ -> N, Val(DIM))..., E, N, DIM)
    U = dgsem_node(u, Val(NV), node..., e)
    @unroll for d in 1:DIM
        n = dgsem_metric(Ja, Val(DIM), d, node..., e)
        for l in (node[d] + 1):N
            other = dgsem_line(node, d, l)
            f = dgsem_twopoint_flux(
                U, dgsem_node(u, Val(NV), other..., e),
                dgsem_average(n, dgsem_metric(Ja, Val(DIM), d, other..., e))
            )
            @unroll for v in 1:NV
                F[dgsem_local_index(layout, v, node..., le, l, d)...] = f[v]
            end
        end
    end
    @synchronize
    acc = dgsem_zero(eltype(du), Val(NV))
    @unroll for d in 1:DIM
        for l in 1:N
            if l < node[d]
                f = dgsem_local_node(F, layout, Val(NV), dgsem_line(node, d, l)..., le, node[d], d)
                acc = muladd.(Ds[node[d], l], f, acc)
            elseif l > node[d]
                f = dgsem_local_node(F, layout, Val(NV), node..., le, l, d)
                acc = muladd.(Ds[node[d], l], f, acc)
            end
        end
    end
    @unroll for v in 1:NV
        du[v, node..., e] = acc[v]
    end
end

# ------------------------------------------------------------------------------ driver

# Random data for `nelem` elements with `N` nodes per direction: a state with positive
# density and pressure, metric terms close to the identity and derivative matrices (`Ds`
# with a zero diagonal).
function dgsem_data(backend, ::Type{T}, ::Val{DIM}, N, nelem) where {T, DIM}
    rng = Random.Xoshiro(42)
    NV = DIM + 2
    nodes = ntuple(_ -> N, DIM)
    u = rand(rng, T, NV, nodes..., nelem)
    u[1, ntuple(_ -> :, DIM + 1)...] .+= 1
    u[2:(DIM + 1), ntuple(_ -> :, DIM + 1)...] .-= T(0.5)
    u[NV, ntuple(_ -> :, DIM + 1)...] .+= 10
    Ja = T(0.1) .* rand(rng, T, DIM, DIM, nodes..., nelem)
    for d in 1:DIM
        Ja[d, d, ntuple(_ -> :, DIM + 1)...] .+= 1
    end
    D = randn(rng, T, N, N)
    Ds = randn(rng, T, N, N)
    for i in 1:N
        Ds[i, i] = 0
    end
    device(A) = copyto!(KernelAbstractions.allocate(backend, T, size(A)), A)
    return (; u = device(u), Ja = device(Ja), D = device(D), Ds = device(Ds))
end

# Work-items per workgroup of the kernels with one work-item per element
const DGSEM_ELEMENT_WORKGROUPSIZE = 64

# The variants for `DIM` dimensions, as `name => f!(du, data)`
function dgsem_variants(backend, ::Val{DIM}, N, nelem) where {DIM}
    nodes = ntuple(_ -> N, DIM)
    # elements per workgroup such that a workgroup has 128 work-items, as in Trixi
    Ew = max(1, 128 ÷ N^DIM)

    # `matrix(data)` is the derivative matrix of the variant
    per_element(kernel, matrix) = let k = kernel(backend, DGSEM_ELEMENT_WORKGROUPSIZE)
        (du, data) -> k(du, data.u, matrix(data), data.Ja, Val(N), Val(DIM); ndrange = nelem)
    end
    per_node(kernel, matrix, E, args...) = let k = kernel(backend, (nodes..., E))
        (du, data) -> k(du, data.u, matrix(data), data.Ja, Val(N), Val(DIM), args...; ndrange = (nodes..., nelem))
    end
    weak(data) = data.D
    fluxdiff(data) = data.Ds

    variants = Pair{String, Any}[
        "weak/element" => per_element(dgsem_weak_element!, weak),
        "weak/element private" => per_element(dgsem_weak_element_private!, weak),
    ]
    layouts = (DGSEMVariableFirst(), DGSEMVariableLast())
    for E in unique((1, Ew)), layout in layouts
        name = "weak/node localmem$(dgsem_layout_name(backend, layout)) E=$E"
        push!(variants, name => per_node(dgsem_weak_node_localmem!, weak, E, Val(E), layout))
    end
    push!(variants, "weak/node recompute E=$Ew" => per_node(dgsem_weak_node_recompute!, weak, Ew))
    if DIM == 3
        for E in unique((1, max(1, 128 ÷ N^2))), NB in (1, 2), layout in layouts
            k = dgsem_weak_slab_3d!(backend, (N, N, E))
            name = "weak/slab$(NB == 2 ? " double-buffered" : "")$(dgsem_layout_name(backend, layout)) E=$E"
            push!(variants, name => (du, data) -> k(du, data.u, data.D, data.Ja, Val(N), Val(E), Val(NB), layout; ndrange = (N, N, nelem)))
        end
    end
    push!(variants, "fluxdiff/element" => per_element(dgsem_fluxdiff_element!, fluxdiff))
    push!(variants, "fluxdiff/node E=$Ew" => per_node(dgsem_fluxdiff_node!, fluxdiff, Ew))
    # In 3D, `E = 2` would need more than 48 KiB of local memory in Float64
    Eh = DIM == 2 ? Ew : 1
    for layout in layouts
        name = "fluxdiff/node halfsweep$(dgsem_layout_name(backend, layout)) E=$Eh"
        push!(variants, name => per_node(dgsem_fluxdiff_halfsweep!, fluxdiff, Eh, Val(Eh), layout))
    end
    DGSEM_TILES && append!(variants, dgsem_tile_variants(backend, Val(DIM), N, nelem))
    return variants
end

# Variants with tiles, for versions of KernelAbstractions that have them
const DGSEM_TILES = isdefined(KernelAbstractions, :tile_shfl)
DGSEM_TILES && include("dgsem_tiles.jl")

# Small meshes on the CPU, to keep the time of the suite down, and meshes that fill a GPU
# otherwise. All are divisible by the number of elements per workgroup.
const DGSEM_NNODES = 4
const DGSEM_NELEMENTS = BACKEND isa CPU ? (32^2, 8^3) : (256^2, 32^3)

SUITE["dgsem"] = BenchmarkGroup()

let N = DGSEM_NNODES
    for (DIM, nelem) in zip((2, 3), DGSEM_NELEMENTS)
        dims = BenchmarkGroup()
        for T in (Float32, Float64)
            data = dgsem_data(BACKEND, T, Val(DIM), N, nelem)
            du = KernelAbstractions.zeros(BACKEND, T, size(data.u)...)
            group = BenchmarkGroup()
            # A launch takes at least ~100 µs, so a single evaluation per sample is enough,
            # which also skips the tuning. One second per benchmark keeps the group short.
            for (name, f!) in dgsem_variants(BACKEND, Val(DIM), N, nelem)
                group[name] = @benchmarkable begin
                    $f!($du, $data)
                    KernelAbstractions.synchronize($BACKEND)
                end evals = 1 seconds = 1
            end
            dims["$T"] = group
        end
        SUITE["dgsem"]["$(DIM)D"] = dims
    end
end
