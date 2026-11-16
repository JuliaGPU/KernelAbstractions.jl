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
# shared between all of its nodes, so there are several ways to map it to work-items. Each
# of the kernels below stresses something else:
#
# - `dgsem_weak_element!`: one work-item per element, which loops over the nodes and
#   scatters their contributions into `du` in global memory, i.e., read-modify-write. This
#   is the CPU code run per element, which is what Trixi's generic GPU fallback does.
# - `dgsem_weak_node_localmem!`: one work-item per node, workgroups of `E` elements. Each
#   work-item stores its fluxes in `@localmem`, and after a `@synchronize` sums along its
#   lines and writes its `du` once.
# - `dgsem_fluxdiff_node!`: the flux differencing form, with a symmetric two-point flux of
#   the averaged state of two nodes on a line instead of the derivative of the flux. Each
#   work-item computes all the two-point fluxes of its node and writes `du` once.
#
# Local-memory layout: the variables of a node are stored either first, `F[v, i, j, (k,) e,
# ...]`, or last, `F[i, j, (k,) e, ..., v]`. With the variable index last, neighbouring
# work-items access neighbouring words, while with it first and an even number of
# variables (`NV = 4` in 2D) they access the same shared-memory banks on NVIDIA GPUs
# (trixi-framework/Trixi.jl#3337). On the CPU the variables of a node in consecutive words
# are faster. So, like the `VIJFH` and `VIJHF` data layouts of ClimaCore.jl, each backend
# gets a default layout (`dgsem_default_layout`).
#
# The values of `D` and `Ja` don't matter for the timing, so they are random.

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

# `Ds` has a zero diagonal, as Trixi's `derivative_split`
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
