###
# Launch configurations
#
# How a backend maps the hardware groups and work-items of a launch onto the blocked
# iteration space. The launch is stored in the kernel's `CompilerMetadata`, so the index
# functions below can specialize on it.
###

"""
    LinearLaunch{T}()

Launch configuration for a kernel launched on a 1-D grid: the x-components of the hardware
group and local ids are the column-major positions in `blocks(iterspace)` and
`workitems(iterspace)`, like with the default launch (`nothing`).

`@index` computes in `T` (`Int32` or `Int`), which has to hold the number of work-items
in the padded iteration space. It still returns `Int`s.
"""
struct LinearLaunch{T <: Integer} end

"""
    NDLaunch{T}()

Launch configuration for a kernel launched on an N-d grid, where `N = ndims(iterspace)` is
at most the number of grid dimensions of the backend (the length of
[`KI.max_work_group_dims`](@ref KernelInterface.max_work_group_dims), i.e. 3): the grid
consists of `size(blocks(iterspace))` groups of `size(workitems(iterspace))` work-items, so
the x, y and z-components of the hardware group and local ids are the positions along the
first `N` dimensions of the iteration space. This avoids decomposing linear ids into
Cartesian positions, i.e., divisions.

The linear group and local indices are computed x-fastest, so they agree with the ordering
of a [`LinearLaunch`](@ref) (and with how GPUs typically form sub-groups).

`@index` computes in `T` (`Int32` or `Int`), which has to hold the number of work-items
in the padded iteration space. It still returns `Int`s.
"""
struct NDLaunch{T <: Integer} end

const Launch = Union{LinearLaunch, NDLaunch}

index_type(::LinearLaunch{T}) where {T} = T
index_type(::NDLaunch{T}) where {T} = T


## host side

"""
    select_launch(kernel::Kernel, workgroupsize, iterspace)::Union{LinearLaunch, NDLaunch}

Choose how to launch `kernel` with the (possibly preliminary) iteration space `iterspace`
and the given `workgroupsize` (`nothing` if it will be tuned), as returned by
`launch_config`: an [`NDLaunch`](@ref) if the iteration space fits the grid of
`backend(kernel)`, i.e. its number of dimensions and per-dimension limits, and a
[`LinearLaunch`](@ref) otherwise, both computing indices in `Int32` if possible.

If the workgroup size will be tuned, the choice holds for any workgroup size the backend
tunes afterwards, so the context and thus the compiled kernel are the same before and
after tuning. That requires tuning with [`launch_workgroupsize`](@ref) and with at most
`KI.max_work_group_size(backend)` work-items, and assumes that `iterspace` covers the
padded iteration space with a single workgroup, as `launch_config` does.

Throws an `ArgumentError` if the iteration space has more than `typemax(Int)` work-items.

The result is not inferred concretely: launching with it takes a dynamic dispatch, unless
the caller checks for the common case of an `NDLaunch{Int32}` first.
"""
function select_launch(kernel::Kernel, workgroupsize, iterspace)
    b = backend(kernel)
    groups = size(blocks(iterspace))
    items = size(workitems(iterspace))
    tuned = KernelAbstractions.workgroupsize(kernel) <: DynamicSize && workgroupsize === nothing
    return select_launch(
        map(mul_extent, groups, items), tuned ? nothing : items,
        KI.max_work_group_size(b), KI.max_work_group_dims(b), KI.max_num_groups(b)
    )
end

# `groupsize === nothing` means that the workgroup size will be tuned
function select_launch(
        extent::Dims{N}, groupsize, max_items::Int,
        max_dims::Dims, max_groups::Dims
    ) where {N}
    nd = N <= length(max_dims)
    if groupsize === nothing
        if nd
            # a tuned workgroup has at least one work-item per dimension
            nd = all(ntuple(d -> extent[d] <= max_groups[d], Val(N)))
        end
        padded = tuned_padded(extent, max_items, nd ? max_dims : ())
    else
        blocks, items, _ = NDIteration.partition(extent, groupsize)
        if nd
            nd = all(ntuple(d -> items[d] <= max_dims[d] && blocks[d] <= max_groups[d], Val(N))) &&
                prod(items) <= max_items
        end
        padded = map(mul_extent, blocks, items)
    end
    if fits(Int32, padded)
        return nd ? NDLaunch{Int32}() : LinearLaunch{Int32}()
    elseif fits(Int, padded)
        return nd ? NDLaunch{Int}() : LinearLaunch{Int}()
    else
        throw_too_large()
    end
end

# arithmetic on the extents of an iteration space, which has to fit an `Int`
@noinline throw_too_large() =
    throw(ArgumentError("Iteration space has more than typemax(Int) work-items"))
function mul_extent(a::Int, b::Int)
    n, overflow = Base.mul_with_overflow(a, b)
    overflow && throw_too_large()
    return n
end
function add_extent(a::Int, b::Int)
    n, overflow = Base.add_with_overflow(a, b)
    overflow && throw_too_large()
    return n
end

# Upper bound on the padded extents of an iteration space whose workgroup size is chosen
# by `launch_workgroupsize`, i.e., by `KI.threads_to_workgroupsize` with at most `capacity`
# work-items and the per-dimension `limits`. That fills the first dimensions first, so a
# dimension only gets more than one work-item if the previous ones don't use all of them,
# and it pads each dimension by less than its number of work-items.
tuned_padded(::Tuple{}, capacity, limits) = ()
function tuned_padded(extent::Tuple, capacity, limits)
    limit = isempty(limits) ? typemax(Int) : first(limits)
    items = max(1, min(first(extent), capacity, limit))
    padded = iszero(first(extent)) ? 0 : add_extent(first(extent), items - 1)
    rest = isempty(limits) ? () : Base.tail(limits)
    return (padded, tuned_padded(Base.tail(extent), max(1, capacity ÷ items), rest)...)
end

# whether the number of positions in an iteration space with extents `dims` fits `T`
fits(::Type{T}, dims::Dims) where {T} = _fits(T, 1, dims)
_fits(::Type, n, ::Tuple{}) = true
function _fits(::Type{T}, n, dims::Dims) where {T}
    n, overflow = Base.mul_with_overflow(n, first(dims))
    return if overflow || n > typemax(T)
        any(iszero, dims)
    else
        _fits(T, n, Base.tail(dims))
    end
end

"""
    launch_workgroupsize(backend, launch, threads, ndrange)

The workgroup size for launching `threads` work-items per workgroup over `ndrange` with
`launch` (see [`select_launch`](@ref)). Backends that tune the workgroup size of a kernel
launched with a [`LinearLaunch`](@ref) or [`NDLaunch`](@ref) have to use this.
"""
launch_workgroupsize(backend, ::NDLaunch, threads, ndrange) =
    KI.threads_to_workgroupsize(threads, extents(ndrange), KI.max_work_group_dims(backend))
launch_workgroupsize(backend, ::Union{LinearLaunch, Nothing}, threads, ndrange) =
    KI.threads_to_workgroupsize(threads, extents(ndrange))


## device side

# The sizes of the iteration space are stored as `Int`s, but fit the index type `T`
# of the launch, so truncating them is safe. (Written without closures capturing `T`,
# which Julia 1.10 fails to infer.)
@inline narrow(::Type{T}, ::Tuple{}) where {T} = ()
@inline narrow(::Type{T}, dims::Tuple) where {T} = (first(dims) % T, narrow(T, Base.tail(dims))...)

# Widen a (positive) index of type `T` to the `Int` returned by `@index`.
@inline widen_index(i::Int) = i
@inline widen_index(i::T) where {T <: Union{Int8, Int16, Int32}} = (i % unsigned(T)) % Int

# 0-based linear index into `dims` to 1-based subscripts, using unsigned divisions
@inline ind2sub(::Tuple{}, i) = ()
@inline ind2sub(::Tuple{Any}, i) = (i + one(i),)
@inline function ind2sub(dims::Tuple, i)
    q = Core.Intrinsics.udiv_int(i, dims[1])
    return (i - q * dims[1] + one(i), ind2sub(Base.tail(dims), q)...)
end

# 1-based subscripts into `dims` to a 1-based linear index of type `T`, column-major
@inline linearize(::Type{T}, ::Tuple{}, ::Tuple{}) where {T} = one(T)
@inline linearize(::Type{T}, dims::Tuple, I::Tuple) where {T} =
    I[1] + dims[1] * (linearize(T, Base.tail(dims), Base.tail(I)) - one(T))

@inline hardware_position(id, ::Val{N}) where {N} = ntuple(d -> (id.x, id.y, id.z)[d], Val(N))

# The hardware ids are bounded by the launch, which e.g. lets the compiler fold the validity
# check of a statically-sized kernel.
@inline function assume_bounded(pos::Tuple, dims::Tuple)
    map(pos, dims) do p, d
        NDIteration.assume((p >= one(p)) & (p <= d))
    end
    return pos
end
@inline function assume_bounded(i, dims::Tuple)
    NDIteration.assume((i >= zero(i)) & (i < prod(dims)))
    return i
end

# 1-based position of the current group in `blocks(iterspace)`, and of the current
# work-item in `workitems(iterspace)`, as tuples of the index type
@inline function group_position(ctx, ::NDLaunch{T}) where {T}
    dims = narrow(T, size(blocks(__iterspace(ctx))))
    return assume_bounded(hardware_position(KI.get_group_id(T), Val(ndims(ctx))), dims)
end
@inline function local_position(ctx, ::NDLaunch{T}) where {T}
    dims = narrow(T, size(workitems(__iterspace(ctx))))
    return assume_bounded(hardware_position(KI.get_local_id(T), Val(ndims(ctx))), dims)
end
@inline function group_position(ctx, ::LinearLaunch{T}) where {T}
    dims = narrow(T, size(blocks(__iterspace(ctx))))
    return ind2sub(dims, assume_bounded(KI.get_group_id(T).x - one(T), dims))
end
@inline function local_position(ctx, ::LinearLaunch{T}) where {T}
    dims = narrow(T, size(workitems(__iterspace(ctx))))
    return ind2sub(dims, assume_bounded(KI.get_local_id(T).x - one(T), dims))
end

# 1-based position of the current work-item in the padded iteration space
@inline function blocked_position(ctx, launch::Launch)
    T = index_type(launch)
    groupsize = narrow(T, size(workitems(__iterspace(ctx))))
    return map(
        (g, w, l) -> (g - one(g)) * w + l,
        group_position(ctx, launch), groupsize, local_position(ctx, launch)
    )
end

# The group and work-item indices, i.e. the elements of `blocks(iterspace)` and
# `workitems(iterspace)` at the current positions. These are usually 1-based.
@inline group_index(ctx, launch::Launch) =
    CartesianIndex(offset_position(group_position(ctx, launch), blocks(__iterspace(ctx))))
@inline local_index(ctx, launch::Launch) =
    CartesianIndex(offset_position(local_position(ctx, launch), workitems(__iterspace(ctx))))
@inline offset_position(pos::Tuple, indices::CartesianIndices) =
    map((p, f) -> widen_index(p) + f - 1, pos, first(indices).I)

# For the iteration spaces KernelAbstractions creates itself (1-based blocks and
# work-items, and an identity or offset mapping), the global index is the blocked position
# shifted by the offsets. Other iteration spaces (e.g. with a custom mapping, see #781,
# or a custom `expand` for other contents) have to go through `expand`, `in` and
# `linear_index`.
const OneBasedIndices{N} = CartesianIndices{N, <:NTuple{N, Base.OneTo}}
const BuiltinNDRange = NDRange{
    N, B, W, <:Union{Nothing, OneBasedIndices}, <:Union{Nothing, OneBasedIndices},
    <:Union{Nothing, StaticOffset, DynamicOffset},
} where {N, B, W}
@inline builtin(iterspace::BuiltinNDRange) =
    blocks(iterspace) isa OneBasedIndices && workitems(iterspace) isa OneBasedIndices
@inline builtin(iterspace) = false

@inline global_cartesian(ctx, launch::Launch, iterspace, ndrange) =
if builtin(iterspace) && ndrange isa CartesianIndices
    I = map((i, o) -> widen_index(i) + o, blocked_position(ctx, launch), offsets(iterspace))
    CartesianIndex(I)
else
    @inbounds expand(iterspace, group_index(ctx, launch), local_index(ctx, launch))
end

@inline global_linear(ctx, launch::Launch, iterspace, ndrange) =
if builtin(iterspace) && ndrange isa CartesianIndices
    T = index_type(launch)
    widen_index(linearize(T, narrow(T, size(ndrange)), blocked_position(ctx, launch)))
else
    linear_index(iterspace, ndrange, group_index(ctx, launch), local_index(ctx, launch))
end

@inline validindex(ctx, launch::Launch, iterspace, ndrange) =
if builtin(iterspace) && ndrange isa CartesianIndices
    T = index_type(launch)
    all(map(<=, blocked_position(ctx, launch), narrow(T, size(ndrange))))
else
    global_cartesian(ctx, launch, iterspace, ndrange) in ndrange
end

# `@index` entry points, see `__index_*`
@inline local_linear(ctx, ::LinearLaunch{T}) where {T} = widen_index(KI.get_local_id(T).x)
@inline group_linear(ctx, ::LinearLaunch{T}) where {T} = widen_index(KI.get_group_id(T).x)
@inline local_linear(ctx, launch::NDLaunch{T}) where {T} = widen_index(
    linearize(T, narrow(T, size(workitems(__iterspace(ctx)))), local_position(ctx, launch))
)
@inline group_linear(ctx, launch::NDLaunch{T}) where {T} = widen_index(
    linearize(T, narrow(T, size(blocks(__iterspace(ctx)))), group_position(ctx, launch))
)
@inline local_cartesian(ctx, launch::Launch) = local_index(ctx, launch)
@inline group_cartesian(ctx, launch::Launch) = group_index(ctx, launch)
@inline global_cartesian(ctx, launch::Launch) =
    global_cartesian(ctx, launch, __iterspace(ctx), __ndrange(ctx))
@inline global_linear(ctx, launch::Launch) =
    global_linear(ctx, launch, __iterspace(ctx), __ndrange(ctx))
@inline validindex(ctx, launch::Launch) =
    validindex(ctx, launch, __iterspace(ctx), __ndrange(ctx))
