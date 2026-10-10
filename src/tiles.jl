###
# Tiles
# - @tile
# - tile_shfl, tile_shfl_up, tile_shfl_down, tile_shfl_xor
# - tile_any, tile_all, tile_ballot, tile_reduce, tile_barrier
###

export @tile, Tile, tiles_per_workgroup
export tile_shfl, tile_shfl_up, tile_shfl_down, tile_shfl_xor
export tile_any, tile_all, tile_ballot, tile_reduce, tile_barrier

"""
    Tile{N}

The tile of `N` work-items a work-item belongs to, in a kernel with tiles
(`@kernel tile=N`), as returned by [`@tile`](@ref): `t.index` is the index of the tile within
the workgroup, and `t.lane` the index of the work-item within the tile, both 1-based.
"""
struct Tile{N}
    index::Int
    lane::Int
end

"""
    @tile()::Tile{N}

The tile of the work-item, in a kernel with tiles of `N` work-items:

```julia
@kernel tile=32 function tile_sums!(out, @Const(x))
    t = @tile()
    v = tile_reduce(t, +, x[@index(Global, Linear)])
    if t.lane == 1
        out[@index(Tile)] = v
    end
end
tile_sums!(backend)(out, x; ndrange = 32 * ntiles)
```

The tiles divide the workgroup by its linear local index: the work-item with
`@index(Local, Linear) == lid` is in tile `fld(lid - 1, N) + 1`, at lane `mod(lid - 1, N) + 1`.
[`@index(Tile)`](@ref @index) is the global linear index of the tile,
`(@index(Group, Linear) - 1) * (@groupsize()[1] ÷ N) + t.index`.

The work-items of a tile communicate with the tile operations, which take the tile as their
first argument, so that helper functions can use them: [`tile_shfl`](@ref),
[`tile_shfl_up`](@ref), [`tile_shfl_down`](@ref), [`tile_shfl_xor`](@ref),
[`tile_any`](@ref), [`tile_all`](@ref), [`tile_ballot`](@ref), [`tile_reduce`](@ref) and
[`tile_barrier`](@ref). All work-items of a tile have to execute them together, in control
flow that is uniform over the tile, but can differ between tiles: e.g. a tile can `return`
early, or loop a different number of times. [`@synchronize`](@ref) and the work-group
collectives still need the whole workgroup.

Kernels with tiles have restrictions on their launch:

- `N` is a power of two of at most the sub-group width of the backend, and the backend has
  to support tiles of `N` work-items, see [`tiles_per_workgroup`](@ref);
- the `ndrange` is 1-D, and a multiple of `N`, so that work-items that pad a partial
  workgroup form whole tiles, which don't execute the kernel;
- the workgroup size is a multiple of `N`, and equal to `N` on backends that only support one
  tile per workgroup. Without an explicit workgroup size, KernelAbstractions picks one that
  satisfies this; an explicit one that doesn't is an error.
"""
macro tile()
    return :($__tile($(esc(:__ctx__)), $(esc(:__tile_width__))))
end

@inline function __tile(ctx, ::Val{N}) where {N}
    lid0 = __index_Local_Linear(ctx) - 1
    return Tile{N}(lid0 ÷ N + 1, lid0 % N + 1)
end

@inline function __index_Tile(ctx, ::Val{N}) where {N}
    lid0 = __index_Local_Linear(ctx) - 1
    return (__index_Group_Linear(ctx) - 1) * (groupsize(ctx)[1] ÷ N) + lid0 ÷ N + 1
end

# A tile is a sub-group, or the work-group that is a single (possibly partial) sub-group of `N`
# work-items, so the tile operations are KernelInterface's sub-group operations; the shuffles
# use a `width` of `N`, so that they don't read past the tile.

"""
    tile_shfl(t::Tile{N}, val, lane)

`val` of the work-item at lane `lane` of the tile `t` (taken modulo `N`). See
[`KernelInterface.shfl`](@ref) for the supported types.
"""
@inline tile_shfl(::Tile{N}, val, lane::Integer) where {N} = KI.shfl(val, lane, N)

"""
    tile_shfl_up(t::Tile{N}, val, offset)

`val` of the work-item `offset` lanes earlier in the tile `t`, or the work-item's own `val`
if there is none. See [`KernelInterface.shfl`](@ref) for the supported types.
"""
@inline tile_shfl_up(::Tile{N}, val, offset::Integer) where {N} = KI.shfl_up(val, offset, N)

"""
    tile_shfl_down(t::Tile{N}, val, offset)

`val` of the work-item `offset` lanes further in the tile `t`, or the work-item's own `val`
if there is none. See [`KernelInterface.shfl`](@ref) for the supported types.
"""
@inline tile_shfl_down(::Tile{N}, val, offset::Integer) where {N} = KI.shfl_down(val, offset, N)

"""
    tile_shfl_xor(t::Tile{N}, val, mask)

`val` of the work-item at the lane whose 0-based index is the work-item's xor `mask`, which
has to be between 0 and `N - 1`. See [`KernelInterface.shfl`](@ref) for the supported types.
"""
@inline tile_shfl_xor(::Tile{N}, val, mask::Integer) where {N} = KI.shfl_xor(val, mask, N)

"""
    tile_any(t::Tile, pred::Bool)::Bool

Whether `pred` is `true` for any work-item of the tile `t`.
"""
@inline tile_any(::Tile, pred::Bool) = KI.sub_group_any(pred)

"""
    tile_all(t::Tile, pred::Bool)::Bool

Whether `pred` is `true` for all work-items of the tile `t`.
"""
@inline tile_all(::Tile, pred::Bool) = KI.sub_group_all(pred)

"""
    tile_ballot(t::Tile, pred::Bool)::UInt64

A mask of the work-items of the tile `t` for which `pred` is `true`: bit `i - 1` for the
work-item at lane `i`. Needs a sub-group width of at most 64
([`KernelInterface.sub_group_size`](@ref)).
"""
@inline tile_ballot(::Tile, pred::Bool) = KI.sub_group_ballot(pred)

"""
    tile_reduce(t::Tile, op, val)

Reduce `val` over the work-items of the tile `t` with the associative operator `op`, in the
order of the lanes, and return the result on every work-item of the tile. See
[`KernelInterface.sub_group_reduce`](@ref).
"""
@inline tile_reduce(::Tile, op, val) = KI.sub_group_reduce(op, val)

"""
    tile_barrier(t::Tile)

Wait until all work-items of the tile `t` reached the barrier, and make their writes to
global and local memory before it visible to the tile.
"""
@inline tile_barrier(::Tile) = KI.sub_group_barrier()


## host side

"""
    tiles_per_workgroup(backend, N)::Int

How many tiles of `N` work-items a workgroup of a kernel with tiles (`@kernel tile=N`) can
have on `backend`: `0` if the backend doesn't support such tiles, `1` if a workgroup has to
be a single tile, or `typemax(Int)` if it can have several (up to the kernel's workgroup size
limit, which the launch checks).

Tiles need sub-groups that are formed from consecutive work-items
([`KernelInterface.supports_linear_subgroups`](@ref)), and `N` a power of two of at most the
sub-group width `W`. Several tiles per workgroup need tiles that are sub-groups (`N == W`)
which can communicate independently
([`KernelInterface.supports_independent_subgroups`](@ref)).
"""
function tiles_per_workgroup(backend::KI.Backend, N::Integer)
    (N > 0 && ispow2(N) && KI.supports_linear_subgroups(backend)) || return 0
    W = KI.sub_group_size(backend)
    N <= W || return 0
    return N == W && KI.supports_independent_subgroups(backend) ? typemax(Int) : 1
end

# The tile width of a kernel with tiles, or `nothing`: the `@kernel` macro adds a method for
# the kernel's function.
tile_width(f) = nothing

tile_error(N, msg) = throw(ArgumentError("Invalid launch of a kernel with tiles of $N work-items: $msg"))

# The workgroup size to launch a kernel with tiles of `N` work-items with, without an
# explicit one: `N` if a workgroup has to be a single tile, or `nothing` to tune it.
function tile_workgroupsize(kernel::Kernel, N, workgroupsize)
    workgroupsize === nothing && KernelAbstractions.workgroupsize(kernel) <: DynamicSize || return workgroupsize
    tiles = tiles_per_workgroup(backend(kernel), N)
    tiles == 0 && tile_error(N, "the backend doesn't support them, see `tiles_per_workgroup`")
    return tiles == 1 ? N : nothing
end

# A tuned workgroup size for a kernel with tiles of `N` work-items: a multiple of `N`
tile_threads(N, threads) = max(N, threads ÷ N * N)

# Check the launch of `kernel` with tiles of `N` work-items with the `ndrange`, `workgroupsize`
# and `iterspace` from `launch_config`: the workgroup size is the partitioned one, or a
# multiple of `N` if it will be tuned.
function check_tiles_launch(kernel::Kernel, N, ndrange, workgroupsize, iterspace)
    range = something(ndrange, static_ndrange(kernel))
    tuned = workgroupsize === nothing && KernelAbstractions.workgroupsize(kernel) <: DynamicSize
    items = tuned ? (N,) : size(workitems(iterspace))
    return check_tiles(backend(kernel), N, extents(range), KI.pad3(items))
end

# Check the launch of a kernel with tiles of `N` work-items over `ndrange`, in workgroups of
# `workgroupsize`
function check_tiles(backend, N, ndrange::Dims, workgroupsize::Dims)
    tiles = tiles_per_workgroup(backend, N)
    tiles == 0 && tile_error(N, "the backend doesn't support them, see `tiles_per_workgroup`")
    length(ndrange) == 1 || tile_error(N, "the ndrange $ndrange isn't 1-D")
    ndrange[1] % N == 0 || tile_error(N, "the ndrange $(ndrange[1]) isn't a multiple of $N")
    all(==(1), Base.tail(workgroupsize)) ||
        tile_error(N, "the workgroup size $workgroupsize isn't 1-D")
    items = workgroupsize[1]
    items % N == 0 || tile_error(N, "the workgroup size $items isn't a multiple of $N")
    tiles == 1 && items != N &&
        tile_error(N, "the backend only supports a single tile per workgroup, not a workgroup size of $items")
    return
end
