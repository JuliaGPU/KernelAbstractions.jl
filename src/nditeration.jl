module NDIteration

export _Size, StaticSize, DynamicSize, get
export NDRange, blocks, workitems, expand
export StaticOffset, DynamicOffset, offsets, extents, linear_index
export DynamicCheck, NoDynamicCheck

import Base.@pure

struct DynamicCheck end
struct NoDynamicCheck end

# An axis of an `ndrange` is either an extent (`Int`) or a range of indices (`UnitRange{Int}`).
axis(n::Integer) = Int(n)
axis(r::Base.OneTo) = Int(length(r))
axis(r::AbstractUnitRange) = UnitRange{Int}(r)

extent(n::Integer) = Int(n)
extent(r::AbstractUnitRange) = length(r)

axis_offset(::Integer) = 0
axis_offset(r::AbstractUnitRange) = first(r) - 1

"""
    extents(ndrange)

Number of indices along each axis of `ndrange`, given as a tuple of extents and/or ranges,
a `CartesianIndices`, a single range, or an integer.
"""
extents(t::Tuple) = map(extent, t)
extents(ci::CartesianIndices) = size(ci)
extents(r::AbstractUnitRange) = (length(r),)
extents(n::Integer) = (Int(n),)

"""
    offsets(ndrange)

Offset of the first index along each axis of `ndrange` relative to 1.
"""
offsets(t::Tuple) = map(axis_offset, t)

"""
    normalize_ndrange(ndrange)

Canonical form of a launch `ndrange`: `nothing`, or a tuple of `Int` extents and
`UnitRange{Int}` axes.
"""
normalize_ndrange(::Nothing) = nothing
normalize_ndrange(n::Integer) = (Int(n),)
normalize_ndrange(r::AbstractUnitRange) = (axis(r),)
normalize_ndrange(ci::CartesianIndices) = map(axis, ci.indices)
normalize_ndrange(t::Tuple) = map(axis, t)

"""
    normalize_workgroupsize(workgroupsize)

Canonical form of a launch `workgroupsize`: `nothing`, or a tuple of `Int` extents.
"""
normalize_workgroupsize(::Nothing) = nothing
normalize_workgroupsize(n::Integer) = (Int(n),)
normalize_workgroupsize(t::Tuple) = extents(t)

# Two ndranges denote the same indices.
same_axes(a::Tuple, b::Tuple) = extents(a) == extents(b) && offsets(a) == offsets(b)

"""
    linear_index(ndrange::CartesianIndices, I::CartesianIndex)

Column-major position of `I` within `ndrange`, counted from 1.
"""
@inline function linear_index(ndrange::CartesianIndices{N}, I::CartesianIndex{N}) where {N}
    lo = map(first, ndrange.indices)
    sz = size(ndrange)
    idx = I.I[N] - lo[N]
    for d in (N - 1):-1:1
        idx = idx * sz[d] + (I.I[d] - lo[d])
    end
    return idx + 1
end

abstract type _Size end

"""
    DynamicSize

Marker type indicating that a kernel's workgroup size or `ndrange` is chosen at launch time.
"""
struct DynamicSize <: _Size end

"""
    StaticSize{S}

Marker type encoding a compile-time workgroup size or `ndrange` as a tuple `S`.
Each entry of `S` is an `Int` extent or, for an `ndrange` axis whose indices do not start
at 1, a `UnitRange{Int}`.
"""
struct StaticSize{S} <: _Size
    function StaticSize{S}() where {S}
        return new{S::Tuple{Vararg{Union{Int, UnitRange{Int}}}}}()
    end
end

@pure StaticSize(s::Tuple{Vararg{Int}}) = StaticSize{s}()
@pure StaticSize(s::Int...) = StaticSize{s}()
@pure StaticSize(s::Type{<:Tuple}) = StaticSize{tuple(s.parameters...)}()
StaticSize(s::Tuple{Vararg{Union{Integer, AbstractUnitRange{<:Integer}}}}) = StaticSize{map(axis, s)}()
StaticSize(ci::CartesianIndices) = StaticSize(ci.indices)

# Some @pure convenience functions for `StaticSize`
@pure get(::Type{StaticSize{S}}) where {S} = S
@pure get(::StaticSize{S}) where {S} = S
@pure Base.getindex(::StaticSize{S}, i::Int) where {S} = i <= length(S) ? S[i] : 1
@pure Base.ndims(::StaticSize{S}) where {S} = length(S)
@pure Base.length(::StaticSize{S}) where {S} = prod(extents(S))

"""
    StaticOffset{O}

Compile-time offset `O::NTuple{N, Int}` added to the indices produced by an [`NDRange`](@ref).
"""
struct StaticOffset{O}
    function StaticOffset{O}() where {O}
        return new{O::Tuple{Vararg{Int}}}()
    end
end

"""
    DynamicOffset{N}

Runtime offset added to the indices produced by an [`NDRange`](@ref).
"""
struct DynamicOffset{N}
    offset::NTuple{N, Int}
end

"""
    NDRange

Encodes a blocked iteration space. The `mapping` field relates blocked indices to
`ndrange` indices: `nothing` for the identity, or a [`StaticOffset`](@ref)/[`DynamicOffset`](@ref)
for an `ndrange` whose indices do not start at 1.

# Example
```
ndrange = NDRange{2, DynamicSize, DynamicSize}(CartesianIndices((256, 256)), CartesianIndices((32, 32)))
for block in ndrange
    for items in workitems(ndrange)
        I = expand(ndrange, block, items)
        checkbounds(Bool, A, I) || continue
        @inbounds A[I] = 2*A[I]
    end
end
```
"""
struct NDRange{N, StaticBlocks, StaticWorkitems, DynamicBlock, DynamicWorkitems, Mapping}
    blocks::DynamicBlock
    workitems::DynamicWorkitems
    mapping::Mapping

    function NDRange{N, B, W}() where {N, B, W}
        return new{N, B, W, Nothing, Nothing, Nothing}(nothing, nothing, nothing)
    end

    function NDRange{N, B, W}(blocks, workitems, mapping = nothing) where {N, B, W}
        return new{N, B, W, typeof(blocks), typeof(workitems), typeof(mapping)}(blocks, workitems, mapping)
    end
end

@inline workitems(range::NDRange{N, B, W}) where {N, B, W <: DynamicSize} = range.workitems::CartesianIndices{N}
@inline workitems(range::NDRange{N, B, W}) where {N, B, W <: StaticSize} = CartesianIndices(get(W))::CartesianIndices{N}
@inline blocks(range::NDRange{N, B}) where {N, B <: DynamicSize} = range.blocks::CartesianIndices{N}
@inline blocks(range::NDRange{N, B}) where {N, B <: StaticSize} = CartesianIndices(get(B))::CartesianIndices{N}
@inline Base.ndims(::NDRange{N}) where {N} = N

@inline offsets(::NDRange{N, B, W, DB, DW, Nothing}) where {N, B, W, DB, DW} = ntuple(_ -> 0, Val(N))
@inline offsets(::NDRange{N, B, W, DB, DW, StaticOffset{O}}) where {N, B, W, DB, DW, O} = O
@inline offsets(range::NDRange{N, B, W, DB, DW, DynamicOffset{N}}) where {N, B, W, DB, DW} = range.mapping.offset

# Mapping of a partitioned `ndrange` (in canonical form); a plain size tuple has no mapping.
static_mapping(::Tuple{Vararg{Int}}) = nothing
static_mapping(t::Tuple) = StaticOffset{offsets(t)}()
dynamic_mapping(::Tuple{Vararg{Int}}) = nothing
dynamic_mapping(t::Tuple) = DynamicOffset(offsets(t))

import Base.iterate
@inline iterate(range::NDRange) = iterate(blocks(range))
@inline iterate(range::NDRange, state) = iterate(blocks(range), state)

Base.length(range::NDRange) = length(blocks(range))

@inline function expand(ndrange::NDRange{N}, groupidx::CartesianIndex{N}, idx::CartesianIndex{N}) where {N}
    offset = offsets(ndrange)
    nI = ntuple(Val(N)) do I
        Base.@_inline_meta
        stride = size(workitems(ndrange), I)
        gidx = groupidx.I[I]
        (gidx - 1) * stride + idx.I[I] + offset[I]
    end
    return CartesianIndex(nI)
end


"""
    assume(cond::Bool)

Assume that the condition `cond` is true. This is a hint to the compiler, possibly enabling
it to optimize more aggressively.
"""
@inline assume(cond::Bool) = Base.llvmcall(
    (
        """
        declare void @llvm.assume(i1)

        define void @entry(i8) #0 {
            %cond = icmp eq i8 %0, 1
            call void @llvm.assume(i1 %cond)
            ret void
        }

        attributes #0 = { alwaysinline }""", "entry",
    ),
    Nothing, Tuple{Bool}, cond
)

@inline function assume_nonzero(CI::CartesianIndices)
    return ntuple(Val(ndims(CI))) do I
        Base.@_inline_meta
        indices = CI.indices[I]
        assume(indices.stop > 0)
    end
end

Base.@propagate_inbounds function expand(ndrange::NDRange, groupidx::Integer, idx::Integer)
    # this causes a exception branch and a div
    B = blocks(ndrange)
    W = workitems(ndrange)
    assume_nonzero(B)
    assume_nonzero(W)
    assume(groupidx >= 1)
    assume(groupidx <= length(B))
    assume(idx >= 1)
    assume(idx <= length(W))
    return expand(ndrange, B[groupidx], workitems(ndrange)[idx])
end

Base.@propagate_inbounds function expand(ndrange::NDRange{N}, groupidx::CartesianIndex{N}, idx::Integer) where {N}
    return expand(ndrange, groupidx, workitems(ndrange)[idx])
end

Base.@propagate_inbounds function expand(ndrange::NDRange{N}, groupidx::Integer, idx::CartesianIndex{N}) where {N}
    return expand(ndrange, blocks(ndrange)[groupidx], idx)
end

"""
    partition(ndrange, workgroupsize)

Splits the maximum size of the iteration space by the workgroupsize.
Returns the number of workgroups necessary and whether the last workgroup
needs to perform dynamic bounds-checking.
"""
@inline function partition(ndrange, __workgroupsize)
    ndrange = extents(ndrange)
    __workgroupsize = extents(__workgroupsize)
    @assert length(__workgroupsize) <= length(ndrange)
    # pad workgroupsize with ones
    workgroupsize = ntuple(Val(length(ndrange))) do I
        Base.@_inline_meta
        if I > length(__workgroupsize) || __workgroupsize[I] == 0
            return 1
        else
            return __workgroupsize[I]
        end
    end
    let workgroupsize = workgroupsize
        dynamic = Ref(false)
        blocks = ntuple(Val(length(ndrange))) do I
            Base.@_inline_meta
            dynamic[] |= mod(ndrange[I], workgroupsize[I]) != 0
            return fld1(ndrange[I], workgroupsize[I])
        end

        return blocks, workgroupsize, dynamic[] ? DynamicCheck() : NoDynamicCheck()
    end
end

end #module
