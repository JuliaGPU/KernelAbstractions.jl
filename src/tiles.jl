###
# Tiles
# - @tile
# - tile_shfl, tile_shfl_up, tile_shfl_down, tile_shfl_xor
# - tile_any, tile_all, tile_ballot, tile_reduce, tile_scan, tile_exclusive_scan, tile_barrier
###

export @tile, Tile, tile_width, tiles_per_workgroup
export tile_shfl, tile_shfl_up, tile_shfl_down, tile_shfl_xor
export tile_any, tile_all, tile_ballot, tile_reduce, tile_scan, tile_exclusive_scan, tile_barrier

"""
    Tile{N}

The tile of `N` work-items a work-item belongs to, in a kernel with tiles, as returned by
[`@tile`](@ref): `t.index` is the index of the tile within the workgroup, and `t.lane` the
index of the work-item within the tile, both 1-based. [`tile_width(t)`](@ref tile_width) is
`N`.
"""
struct Tile{N}
    index::Int
    lane::Int
end

"""
    @tile()::Tile{N}

The tile of the work-item, in a kernel with tiles of `N` work-items. A kernel has tiles if it
is declared with `@kernel tile` (the width is chosen when the kernel is constructed) or
`@kernel tile=N` (a fixed width), see "Choosing the tile width" below.

```julia
@kernel tile function tile_sums!(out, @Const(x))
    t = @tile()
    v = tile_reduce(t, +, x[@index(Global, Linear)])
    if t.lane == 1
        out[@index(Tile)] = v
    end
end
kernel = tile_sums!(backend)                # tiles of the sub-group width
N = tile_width(kernel)
kernel(out, x; ndrange = N * ntiles)
```

The tiles divide the workgroup by its linear local index: the work-item with
`@index(Local, Linear) == lid` is in tile `fld(lid - 1, N) + 1`, at lane `mod(lid - 1, N) + 1`.
[`@index(Tile)`](@ref @index) is the global linear index of the tile,
`(@index(Group, Linear) - 1) * (prod(@groupsize()) ÷ N) + t.index`: the tiles are numbered
workgroup by workgroup.

The work-items of a tile communicate with the tile operations, which take the tile as their
first argument, so that helper functions can use them: [`tile_shfl`](@ref),
[`tile_shfl_up`](@ref), [`tile_shfl_down`](@ref), [`tile_shfl_xor`](@ref),
[`tile_any`](@ref), [`tile_all`](@ref), [`tile_ballot`](@ref), [`tile_reduce`](@ref),
[`tile_scan`](@ref), [`tile_exclusive_scan`](@ref) and [`tile_barrier`](@ref). All
work-items of a tile have to execute them together, in control flow that is uniform over the
tile, but can differ between tiles: e.g. a tile can `return` early, or loop a different
number of times. [`@synchronize`](@ref) and the work-group collectives still need the whole
workgroup.

# Choosing the tile width

Tiles are built on the backend's sub-groups, so the width `N` is limited by them: it is a
power of two of at most the sub-group width `W` ([`KernelInterface.sub_group_size`](@ref)),
and the backend has to form sub-groups from consecutive work-items. How many tiles a workgroup
can have depends on the backend, see [`tiles_per_workgroup(backend, N)`](@ref
tiles_per_workgroup): tiles narrower than `W` share a sub-group, which needs a backend whose
sub-groups can be split into independent parts. Where it can't, e.g. on Metal and on the CPU,
a workgroup of such tiles is a single tile. Either way, the tiles behave the same.

- `@kernel tile=N` fixes the width, for algorithms that need it. Constructing the kernel on
  a backend that doesn't support it is an error.
- `@kernel tile` leaves it to the kernel's construction: `f(backend; tile = N)`,
  `f(backend, workgroupsize; tile = N)` or `f(backend, workgroupsize, ndrange; tile = N)`.
  It defaults to the sub-group width `W`, which allows several tiles per workgroup where the
  backend supports them. Code that depends on the width gets it from
  [`tile_width(kernel)`](@ref tile_width) on the host, and `tile_width(t)` in the kernel,
  where it is a constant.

Widths of 32 or less work on all current GPU backends; the sub-group width differs between
them (e.g. 32 on NVIDIA GPUs, 32 or 64 on AMD GPUs, 8 to 32 on Intel GPUs).

# Launching

Kernels with tiles have restrictions on their launch:

- the workgroup has a multiple of `N` work-items, and at most as many tiles as the backend
  supports ([`tiles_per_workgroup`](@ref));
- every tile is either entirely within the `ndrange`, or entirely outside of it, so that the
  work-items that pad a partial workgroup form whole tiles, which don't execute the kernel.
  For a 1-D `ndrange`, that means a multiple of `N`. For an N-D `ndrange`, e.g. one dimension
  per index of an element, it holds if the workgroup covers the leading dimensions of the
  `ndrange` and the number of work-items is a multiple of `N`: e.g. an `ndrange` of
  `(4, 4, nelements)` in workgroups of `(4, 4, k)`, with tiles of 16 work-items or, for an
  even `k` and `nelements`, 32.

Without an explicit workgroup size, KernelAbstractions picks one that satisfies this; an
explicit one that doesn't is an error. Kernels with tiles and an N-D `ndrange` are launched on
a 1-D grid, so that the tiles follow the backend's sub-groups.

To give each element fewer nodes than a tile has lanes, e.g. 9 nodes of a 3×3 element in a
tile of 16, map the lanes to nodes yourself, and keep the lanes without a node in the tile
operations: clamp their loads, skip their stores, but let them shuffle, vote and reduce (with
a neutral value) like the others.
"""
macro tile()
    return :($__tile($(esc(:__ctx__)), $__tile_width($(esc(:__ctx__)))))
end

"""
    tile_width(kernel::Kernel)::Union{Int, Nothing}
    tile_width(t::Tile{N}) = N

The tile width of a kernel with tiles, or `nothing` for a kernel without. In a kernel,
`tile_width(t)` of a tile `t` from [`@tile`](@ref) is a constant.
"""
tile_width(::Kernel{<:Any, <:Any, <:Any, <:Any, T}) where {T} = T
tile_width(::Tile{N}) where {N} = N

# the tile width for the kernel's metadata, see `CompilerMetadata`
tile_val(kernel::Kernel) = (N = tile_width(kernel); N === nothing ? nothing : Val(N))

@inline function __tile(ctx, ::Val{N}) where {N}
    lid0 = __index_Local_Linear(ctx) - 1
    return Tile{N}(lid0 ÷ N + 1, lid0 % N + 1)
end

@inline function __index_Tile(ctx, ::Val{N}) where {N}
    lid0 = __index_Local_Linear(ctx) - 1
    return (__index_Group_Linear(ctx) - 1) * (prod(groupsize(ctx)) ÷ N) + lid0 ÷ N + 1
end

# A tile is a segment of `N` lanes of a sub-group, so the tile operations are KernelInterface's
# sub-group operations with a `width` of `N`: where several tiles share a sub-group, which
# `tiles_per_workgroup` only allows with `KI.supports_independent_segments`, those only need
# the tile to execute them together.

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
@inline tile_any(::Tile{N}, pred::Bool) where {N} = KI.sub_group_any(pred, N)

"""
    tile_all(t::Tile, pred::Bool)::Bool

Whether `pred` is `true` for all work-items of the tile `t`.
"""
@inline tile_all(::Tile{N}, pred::Bool) where {N} = KI.sub_group_all(pred, N)

"""
    tile_ballot(t::Tile, pred::Bool)::UInt64

A mask of the work-items of the tile `t` for which `pred` is `true`: bit `i - 1` for the
work-item at lane `i`. Needs a sub-group width of at most 64
([`KernelInterface.sub_group_size`](@ref)), also for narrower tiles.
"""
@inline tile_ballot(::Tile{N}, pred::Bool) where {N} = KI.sub_group_ballot(pred, N)

"""
    tile_reduce(t::Tile, op, val)

Reduce `val` over the work-items of the tile `t` with the associative operator `op`, in the
order of the lanes, and return the result on every work-item of the tile. See
[`KernelInterface.sub_group_reduce`](@ref).
"""
@inline tile_reduce(::Tile{N}, op, val) where {N} = KI.sub_group_reduce(op, val, N)

"""
    tile_scan(t::Tile, op, val)

The inclusive scan of `val` over the work-items of the tile `t` with the associative
operator `op`, in the order of the lanes. See [`KernelInterface.sub_group_scan`](@ref).
"""
@inline tile_scan(::Tile{N}, op, val) where {N} = KI.sub_group_scan(op, val, N)

"""
    tile_exclusive_scan(t::Tile, op, val, init)

The exclusive scan of `val` over the work-items of the tile `t` with the associative
operator `op`, in the order of the lanes, starting from `init`: lane 1 gets `init`. See
[`KernelInterface.sub_group_exclusive_scan`](@ref).
"""
@inline tile_exclusive_scan(::Tile{N}, op, val, init) where {N} = KI.sub_group_exclusive_scan(op, val, init, N)

"""
    tile_barrier(t::Tile)

Wait until all work-items of the tile `t` reached the barrier, and make their writes to
global and local memory before it visible to the tile. It doesn't order memory between
tiles.
"""
@inline tile_barrier(::Tile{N}) where {N} = KI.sub_group_barrier(N)


## host side

"""
    tiles_per_workgroup(backend, N)::Int

The most tiles of `N` work-items a workgroup of a kernel with tiles (see [`@tile`](@ref)) can
have on `backend`: `0` if the backend doesn't support such tiles, and `typemax(Int)` if it
supports any number (up to the kernel's workgroup size limit, which the launch checks).

Tiles need sub-groups that are formed from consecutive work-items
([`KernelInterface.supports_linear_subgroups`](@ref)), and `N` a power of two of at most the
sub-group width `W`. Then a workgroup can have:

- a tile per sub-group, of any number of sub-groups, if the sub-groups of a workgroup are
  independent ([`KernelInterface.supports_independent_subgroups`](@ref)), or else a single
  sub-group;
- several tiles per sub-group (`N < W`) if its segments of `N` lanes are independent
  ([`KernelInterface.supports_independent_segments`](@ref)), or else a single tile.
"""
function tiles_per_workgroup(backend::KI.Backend, N::Integer)
    (N > 0 && ispow2(N) && KI.supports_linear_subgroups(backend)) || return 0
    W = KI.sub_group_size(backend)
    N <= W || return 0
    segments = N == W || KI.supports_independent_segments(backend)
    if KI.supports_independent_subgroups(backend)
        return segments ? typemax(Int) : 1
    else
        return segments ? W ÷ N : 1
    end
end

# How a kernel function was declared: `:configurable` for `@kernel tile`, the width `N` for
# `@kernel tile=N`, or `missing` for kernels without tiles and other callables. The `@kernel`
# macro adds a method taking a `TilePolicyQuery` to the function of a kernel with tiles, with
# its first method. (`applicable` is constant-folded, so that kernels without tiles don't look
# up the method.)
struct TilePolicyQuery end
tile_policy(f) = applicable(f, TilePolicyQuery()) ? query_tile_policy(f) : missing

# a callable that forwards its arguments, e.g. a wrapper of a kernel function, doesn't declare
# tiles itself
@noinline function query_tile_policy(f)
    sig = Base.unwrap_unionall(which(f, Tuple{TilePolicyQuery}).sig)
    sig.parameters[end] === TilePolicyQuery || return missing
    return f(TilePolicyQuery())
end

# All methods of a kernel function share the tile declaration of the first, since the
# constructors are defined with it: check the declaration `policy` (`nothing` without tiles)
# of a method of `f`, before the first one declares it, and whether a method was defined before.
function declare_tile_policy(f, policy, defined)
    defined || return
    old = coalesce(tile_policy(f), nothing)
    old == policy ||
        error("all methods of a kernel have to be declared with the same tiles, `$(tile_declaration(old))` and `$(tile_declaration(policy))` differ")
    return
end

tile_declaration(::Nothing) = "@kernel"
tile_declaration(policy::Symbol) = "@kernel tile"
tile_declaration(N::Integer) = "@kernel tile=$N"

# Whether a kernel with tiles of width `T` (or `nothing`) can have the function declared
# with `policy`, e.g. for `similar` (other callables than kernel functions, e.g. wrappers, and
# kernels without tiles can't be told apart, so they keep the width)
function check_tile_policy(policy, T)
    if policy === :configurable && T === nothing || policy isa Integer && policy != T
        error("a kernel with tiles of width $T can't run a function declared with `$(tile_declaration(policy))`")
    end
    return
end

# The tile width of a kernel declared with `policy`, for the `tile` keyword of its
# constructor
function resolve_tile_width(backend, policy, tile)
    if policy === nothing || policy === missing
        tile === nothing ||
            throw(ArgumentError("the kernel isn't declared with tiles (`@kernel tile` or `@kernel tile=N`), but got `tile = $tile`"))
        return nothing
    elseif policy === :configurable
        if tile === nothing
            (backend isa KI.Backend && KI.supports_linear_subgroups(backend)) ||
                tile_error_backend()
            return KI.sub_group_size(backend)
        end
        tile isa Integer || throw(ArgumentError("`tile` has to be an integer, got $tile"))
        return Int(tile)
    else
        tile === nothing || tile == policy ||
            throw(ArgumentError("the kernel is declared with tiles of $policy work-items, but got `tile = $tile`"))
        return Int(policy)
    end
end

tile_error_backend() = throw(ArgumentError("the backend doesn't support tiles, as its sub-groups aren't formed from consecutive work-items, see `KernelInterface.supports_linear_subgroups`"))

tile_error(N, msg) = throw(ArgumentError("Tiles of $N work-items: $msg"))

# The workgroup size for a launch of a kernel with tiles of `N` work-items over `range` with
# `launch` (see `select_launch`), for about `threads` work-items: the largest multiple of `N`
# work-items, up to `threads` (but at least `N`, leaving it to the launch to check the kernel's
# limit) and what the backend supports, that divides `range` into whole tiles.
function tile_launch_workgroupsize(backend, launch, N, threads, range)
    dims = extents(range)
    tiles = tiles_per_workgroup(backend, N)
    t = max(threads, N)
    tiles == typemax(Int) || (t = min(t, tiles * N))
    for t in (t ÷ N * N):-N:N
        items = length(dims) == 1 ? launch_workgroupsize(backend, launch, t, range) :
            tile_shape(N, t, dims, index_type(launch))
        items !== nothing && prod(items) % N == 0 && whole_tiles(N, dims, items) && return items
    end
    return tile_error(N, "no workgroup size divides the ndrange $dims into whole tiles, pass one")
end

# The shape of a workgroup of `t` work-items, no larger than `dims` along any dimension, that
# divides `dims` into whole tiles of `N` work-items with the least padding, or `nothing`. The
# padded iteration space has to fit `T`, the index type `select_launch` chose for a workgroup
# from `launch_workgroupsize`, which pads differently.
function tile_shape(N, t, dims::Dims, ::Type{T}) where {T}
    best, best_padded = nothing, typemax(Int)
    for items in workgroup_shapes(t, dims)
        padded = map((d, i) -> Int128(cld(d, i)) * i, dims, items)
        all(<=(typemax(Int)), padded) && fits(T, Int.(padded)) || continue
        n = prod(Int.(padded))
        if n < best_padded && whole_tiles(N, dims, items)
            best, best_padded = items, n
        end
    end
    return best
end

# The shapes of `t` work-items that are no larger than `dims` along any dimension, larger
# extents of the first dimensions first
function workgroup_shapes(t, dims::Dims)
    length(dims) == 1 && return t <= dims[1] ? [(t,)] : Dims{1}[]
    shapes = Dims{length(dims)}[]
    for a in min(t, dims[1]):-1:1
        t % a == 0 || continue
        for rest in workgroup_shapes(t ÷ a, Base.tail(dims))
            push!(shapes, (a, rest...))
        end
    end
    return shapes
end

# Check the launch of `kernel` with tiles of `N` work-items with the `ndrange`, `workgroupsize`
# and `iterspace` from `launch_config`, before compiling it: the partitioned workgroup size, or
# that there is one that will be tuned.
function check_tiles_launch(kernel::Kernel, N, ndrange, workgroupsize, iterspace)
    b = backend(kernel)
    range = something(ndrange, static_ndrange(kernel))
    if workgroupsize === nothing && KernelAbstractions.workgroupsize(kernel) <: DynamicSize
        check_tile_width(b, N)
        check_tile_items(N, extents(range))
        launch = select_launch(kernel, workgroupsize, iterspace)
        tile_launch_workgroupsize(b, launch, N, KI.max_work_group_size(b), range)
        return
    end
    return check_tiles(b, N, extents(range), size(workitems(iterspace)))
end

function check_tile_width(backend, N)
    (N > 0 && ispow2(N)) || tile_error(N, "the width has to be a power of two")
    backend isa KI.Backend && KI.supports_linear_subgroups(backend) || tile_error_backend()
    W = KI.sub_group_size(backend)
    N <= W || tile_error(N, "the width is larger than the sub-group width $W of the backend")
    return
end

check_tile_items(N, ndrange::Dims) = saturated_prod(ndrange) % N == 0 ||
    tile_error(N, "the ndrange $(length(ndrange) == 1 ? only(ndrange) : ndrange) has $(saturated_prod(ndrange)) work-items, which isn't a multiple of $N")

# Check the launch of a kernel with tiles of `N` work-items over `ndrange`, in workgroups of
# `workgroupsize` (logical sizes, before a launch on a 1-D grid flattens them)
function check_tiles(backend, N, ndrange::Dims, workgroupsize::Dims)
    check_tile_width(backend, N)
    check_tile_items(N, ndrange)
    check_tile_workgroup(backend, N, workgroupsize)
    whole_tiles(N, ndrange, workgroupsize) ||
        tile_error(N, "workgroups of $workgroupsize divide the ndrange $ndrange into partial tiles")
    return
end

function check_tile_workgroup(backend, N, workgroupsize::Dims)
    items = prod(workgroupsize)
    items % N == 0 ||
        tile_error(N, "the workgroup size $(length(workgroupsize) == 1 ? items : workgroupsize) isn't a multiple of $N")
    tiles = tiles_per_workgroup(backend, N)
    if items ÷ N > tiles
        tiles == 1 && tile_error(N, "the backend only supports a single tile per workgroup, not a workgroup size of $items")
        tile_error(N, "the backend supports at most $tiles tiles per workgroup, not a workgroup size of $items")
    end
    return
end

# Whether workgroups of size `workgroupsize` (a multiple of `N` work-items) divide `ndrange`
# into tiles of `N` consecutive work-items that are each entirely inside or outside of it. The
# part of a workgroup inside the `ndrange` is a box: the whole workgroup, or in the last
# workgroup along a dimension, what's left of the `ndrange`. So it suffices to check one
# workgroup per combination of the dimensions it's the last one along.
function whole_tiles(N, ndrange::Dims, workgroupsize::Dims)
    n = max(length(ndrange), length(workgroupsize))
    range = ntuple(d -> Base.get(ndrange, d, 1), n)
    items = ntuple(d -> Base.get(workgroupsize, d, 1), n)
    # an empty `ndrange` has no workgroups
    any(iszero, range) && return true
    insides = map(range, items) do r, i
        last = r - (cld(r, i) - 1) * i
        return r > i && last != i ? (i, last) : (last,)
    end
    # a 1-D box, a prefix of the workgroup, is whole tiles if its length is a multiple of `N`
    n == 1 && return all(inside -> inside % N == 0, only(insides))
    return all(inside -> whole_tiles_box(N, items, inside), Iterators.product(insides...))
end

# Whether every tile of `N` consecutive work-items of a workgroup of size `items` is entirely
# inside or outside of the box of size `inside` at its start
function whole_tiles_box(N, items, inside)
    inside == items && return true
    tile_inside = false
    for (l, I) in enumerate(CartesianIndices(items))
        isinside = all(Tuple(I) .<= inside)
        if (l - 1) % N == 0
            tile_inside = isinside
        elseif isinside != tile_inside
            return false
        end
    end
    return true
end
