# Variants of the DGSEM volume integral kernels in `dgsem.jl` with tiles (`@kernel tile`),
# which only exist with KernelAbstractions versions that have tiles. Each tile is an element
# (2D) or a slab of an element (3D), i.e. `N^2` work-items, which exchange values with
# `tile_shfl` instead of `@localmem` and `@synchronize`:
#
# - "weak/node tile E=<E>" (2D): as "weak/node localmem", with the fluxes of the other nodes
#   on a line from `tile_shfl` instead of local memory: no `@localmem` and no barrier.
# - "weak/node localmem tile_barrier E=<E>" (2D): "weak/node localmem" with a
#   `tile_barrier` instead of `@synchronize`, since each element only reads its own local
#   memory.
# - "fluxdiff/node halfsweep tile E=<E>" (2D): each two-point flux is computed once and
#   exchanged with `tile_shfl`, as are the state and the metric terms of the partner node.
#   For uniform control flow over the tile, node `i` computes the fluxes with the nodes
#   `i + o` (cyclically) for `o in 1:N÷2` and gets those with `i - o` from them. For an even
#   `N` the pair at `o = N÷2` is computed by both nodes, with half the weight.
# - "weak/slab tile E=<E>" (3D): "weak/slab" with the fluxes of the slab from `tile_shfl`
#   instead of local memory: no `@localmem` and no barrier.
#
# `E` is the number of elements (or slabs) per workgroup, i.e. tiles per workgroup. Where the
# backend can't have several tiles narrower than its sub-groups in a workgroup (e.g. on the
# CPU), it is 1. The lane of node `(i, j)` in its tile is `i + N * (j - 1)`, since the
# workgroups `(N, N, E)` cover the leading dimensions of the `ndrange`.

@inline dgsem_lane(::Val{N}, i, j) where {N} = i + N * (j - 1)

@kernel inbounds = true tile function dgsem_weak_node_tile_2d!(
        du, @Const(u), @Const(D), @Const(Ja), ::Val{N}
    ) where {N}
    NV = 4
    t = @tile()
    i, j, e = @index(Global, NTuple)
    U = dgsem_node(u, Val(NV), i, j, e)
    G1 = dgsem_flux(U, dgsem_metric(Ja, Val(2), 1, i, j, e))
    G2 = dgsem_flux(U, dgsem_metric(Ja, Val(2), 2, i, j, e))
    acc = dgsem_zero(eltype(du), Val(NV))
    @unroll for l in 1:N
        acc = muladd.(D[i, l], tile_shfl(t, G1, dgsem_lane(Val(N), l, j)), acc)
        acc = muladd.(D[j, l], tile_shfl(t, G2, dgsem_lane(Val(N), i, l)), acc)
    end
    @unroll for v in 1:NV
        du[v, i, j, e] = acc[v]
    end
end

@kernel inbounds = true tile function dgsem_weak_node_tile_barrier_2d!(
        du, @Const(u), @Const(D), @Const(Ja), ::Val{N}, ::Val{E}, layout::DGSEMLayout
    ) where {N, E}
    @uniform NV = 4
    t = @tile()
    i, j, e = @index(Global, NTuple)
    _, _, le = @index(Local, NTuple)
    F = @localmem eltype(du) dgsem_local_index(layout, NV, N, N, E, 2)
    U = dgsem_node(u, Val(NV), i, j, e)
    @unroll for d in 1:2
        G = dgsem_flux(U, dgsem_metric(Ja, Val(2), d, i, j, e))
        @unroll for v in 1:NV
            F[dgsem_local_index(layout, v, i, j, le, d)...] = G[v]
        end
    end
    tile_barrier(t)
    acc = dgsem_zero(eltype(du), Val(NV))
    for l in 1:N
        acc = muladd.(D[i, l], dgsem_local_node(F, layout, Val(NV), l, j, le, 1), acc)
        acc = muladd.(D[j, l], dgsem_local_node(F, layout, Val(NV), i, l, le, 2), acc)
    end
    @unroll for v in 1:NV
        du[v, i, j, e] = acc[v]
    end
end

@kernel inbounds = true tile function dgsem_fluxdiff_halfsweep_tile_2d!(
        du, @Const(u), @Const(Ds), @Const(Ja), ::Val{N}
    ) where {N}
    NV = 4
    T = eltype(du)
    t = @tile()
    i, j, e = @index(Global, NTuple)
    node = (i, j)
    U = dgsem_node(u, Val(NV), i, j, e)
    acc = dgsem_zero(T, Val(NV))
    @unroll for d in 1:2
        n = dgsem_metric(Ja, Val(2), d, i, j, e)
        @unroll for o in 1:(N ÷ 2)
            w = iseven(N) && o == N ÷ 2 ? T(0.5) : one(T)
            after = dgsem_line(node, d, mod1(node[d] + o, N))
            before = dgsem_line(node, d, mod1(node[d] - o, N))
            lane = dgsem_lane(Val(N), after...)
            f = dgsem_twopoint_flux(
                U, tile_shfl(t, U, lane), dgsem_average(n, tile_shfl(t, n, lane))
            )
            # the node `before` computed the flux with this node at the same offset
            fb = tile_shfl(t, f, dgsem_lane(Val(N), before...))
            acc = muladd.(w * Ds[node[d], after[d]], f, acc)
            acc = muladd.(w * Ds[node[d], before[d]], fb, acc)
        end
    end
    @unroll for v in 1:NV
        du[v, i, j, e] = acc[v]
    end
end

# As `dgsem_weak_slab_3d!`, in the same order
@kernel inbounds = true tile function dgsem_weak_slab_tile_3d!(
        du, @Const(u), @Const(D), @Const(Ja), ::Val{N}
    ) where {N}
    @uniform NV = 5
    t = @tile()
    i, j, e = @index(Global, NTuple)
    acc = @private eltype(du) (NV, N)
    @unroll for k in 1:N
        @unroll for v in 1:NV
            acc[v, k] = zero(eltype(du))
        end
    end
    @unroll for k in 1:N
        U = dgsem_node(u, Val(NV), i, j, k, e)
        G1 = dgsem_flux(U, dgsem_metric(Ja, Val(3), 1, i, j, k, e))
        G2 = dgsem_flux(U, dgsem_metric(Ja, Val(3), 2, i, j, k, e))
        G3 = dgsem_flux(U, dgsem_metric(Ja, Val(3), 3, i, j, k, e))
        @unroll for l in 1:N
            Dlk = D[l, k]
            @unroll for v in 1:NV
                acc[v, l] = muladd(Dlk, G3[v], acc[v, l])
            end
        end
        @unroll for l in 1:N
            Dil, Djl = D[i, l], D[j, l]
            F1 = tile_shfl(t, G1, dgsem_lane(Val(N), l, j))
            F2 = tile_shfl(t, G2, dgsem_lane(Val(N), i, l))
            @unroll for v in 1:NV
                acc[v, k] = muladd(Dil, F1[v], acc[v, k])
                acc[v, k] = muladd(Djl, F2[v], acc[v, k])
            end
        end
    end
    @unroll for k in 1:N
        @unroll for v in 1:NV
            du[v, i, j, k, e] = acc[v, k]
        end
    end
end

# The tile variants for `DIM` dimensions, as `name => f!(du, data)`
function dgsem_tile_variants(backend, ::Val{DIM}, N, nelem) where {DIM}
    variants = Pair{String, Any}[]
    width = N^2
    # as many elements per workgroup as fit in 128 work-items and the backend allows; 0 if
    # the backend doesn't support tiles of this width
    E = min(max(1, 128 ÷ width), tiles_per_workgroup(backend, width))
    E > 0 || return variants
    tiled(kernel, matrix, args...) = let k = kernel(backend, (N, N, E); tile = width)
        (du, data) -> k(du, data.u, matrix(data), data.Ja, Val(N), args...; ndrange = (N, N, nelem))
    end
    if DIM == 2
        push!(variants, "weak/node tile E=$E" => tiled(dgsem_weak_node_tile_2d!, data -> data.D))
        for layout in (DGSEMVariableFirst(), DGSEMVariableLast())
            name = "weak/node localmem tile_barrier$(dgsem_layout_name(backend, layout)) E=$E"
            push!(variants, name => tiled(dgsem_weak_node_tile_barrier_2d!, data -> data.D, Val(E), layout))
        end
        push!(variants, "fluxdiff/node halfsweep tile E=$E" => tiled(dgsem_fluxdiff_halfsweep_tile_2d!, data -> data.Ds))
    else
        push!(variants, "weak/slab tile E=$E" => tiled(dgsem_weak_slab_tile_3d!, data -> data.D))
    end
    return variants
end
