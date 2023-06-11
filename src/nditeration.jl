module NDIteration

export _Size, StaticSize, DynamicSize, get
export NDRange, blocks, workitems, expand
export DynamicCheck, NoDynamicCheck

import Base.@pure

struct DynamicCheck end
struct NoDynamicCheck end

abstract type _Size end
struct DynamicSize <: _Size end
struct StaticSize{S} <: _Size
    function StaticSize{S}() where S
        new{S::Tuple{Vararg}}()
    end
end

@pure StaticSize(s::Tuple{Vararg{Int}}) = StaticSize{s}() 
@pure StaticSize(s::Int...) = StaticSize{s}() 
@pure StaticSize(s::Type{<:Tuple}) = StaticSize{tuple(s.parameters...)}()
@pure StaticSize(s::Tuple{Vararg{UnitRange{Int}}}) = StaticSize{s}()

# Some @pure convenience functions for `StaticSize`
@pure get(::Type{StaticSize{S}}) where {S} = S
@pure get(::StaticSize{S}) where {S} = S
@pure Base.getindex(::StaticSize{S}, i::Int) where {S} = i <= length(S) ? S[i] : 1
@pure Base.ndims(::StaticSize{S}) where {S}  = length(S)
@pure Base.length(::StaticSize{S}) where {S} = prod(worksize.(S))

@inline getrange(::StaticSize{S}) where {S} = worksize(S), offsets(S)
@inline getrange(::Type{StaticSize{S}}) where {S} = worksize(S), offsets(S)

@inline worksize(i::Tuple) = worksize.(i)
@inline worksize(i::Int) = i
@inline worksize(i::UnitRange) = length(i)
@inline worksize(i::StepRange) = length(i)

@inline offsets(i) = offsets.(i)
@inline offsets(::NTuple{N, Int}) where N = nothing
@inline offsets(::Int) = nothing
@inline offsets(i::UnitRange) = i.start - 1
@inline offsets(i::StepRange) = i.start - 1, i.step

"""
    NDRange

Encodes a blocked iteration space. 

# Example
```
ndrange = NDRange{2, nothing, DynamicSize, DynamicSize}(CartesianIndices((256, 256)), CartesianIndices((32, 32)))
for block in ndrange
    for items in workitems(ndrange)
        I = expand(ndrange, block, items)
        checkbounds(Bool, A, I) || continue
        @inbounds A[I] = 2*A[I]
    end
end
```
"""
struct NDRange{N, Offsets, StaticBlocks, StaticWorkitems, DynamicBlock, DynamicWorkitems}
    blocks::DynamicBlock
    workitems::DynamicWorkitems

    function NDRange{N, B, W}() where {N, B, W}
        new{N, nothing, B, W, Nothing, Nothing}(nothing, nothing)
    end

    function NDRange{N, B, W}(blocks, workitems) where {N, B, W}
        new{N, nothing, B, W, typeof(blocks), typeof(workitems)}(blocks, workitems)
    end

    function NDRange{N, O, B, W}() where {N, O, B, W}
        new{N, O, B, W, Nothing, Nothing}(nothing, nothing)
    end

    function NDRange{N, O, B, W}(blocks, workitems) where {N, O, B, W}
        new{N, O, B, W, typeof(blocks), typeof(workitems)}(blocks, workitems)
    end
end

@inline workitems(range::NDRange{N, O, B, W}) where {N,O,B,W<:DynamicSize} = range.workitems::CartesianIndices{N}
@inline workitems(range::NDRange{N, O, B, W}) where {N,O,B,W<:StaticSize} = CartesianIndices(get(W))::CartesianIndices{N}
@inline blocks(range::NDRange{N, O, B}) where {N,O,B<:DynamicSize} = range.blocks::CartesianIndices{N}
@inline blocks(range::NDRange{N, O, B}) where {N,O,B<:StaticSize} = CartesianIndices(get(B))::CartesianIndices{N}

import Base.iterate
@inline iterate(range::NDRange) = iterate(blocks(range))
@inline iterate(range::NDRange, state) = iterate(blocks(range), state)

Base.length(range::NDRange) = length(blocks(range))

@inline function expand(ndrange::NDRange{N, nothing}, groupidx::CartesianIndex{N}, idx::CartesianIndex{N}) where N
    nI = ntuple(Val(N)) do I
        Base.@_inline_meta
        stride = size(workitems(ndrange), I)
        gidx = groupidx.I[I]
        (gidx-1)*stride + idx.I[I] 
    end
    CartesianIndex(nI)
end

@inline function expand(ndrange::NDRange{N, Offsets}, groupidx::CartesianIndex{N}, idx::CartesianIndex{N}) where {N, Offsets}
    nI = ntuple(Val(N)) do I
        Base.@_inline_meta
        stride = size(workitems(ndrange), I)
        gidx = groupidx.I[I]
        (gidx-1)*stride + idx.I[I] + Offsets[I]
    end
    CartesianIndex(nI)
end

Base.@propagate_inbounds function expand(ndrange::NDRange, groupidx::Integer, idx::Integer)
    expand(ndrange, blocks(ndrange)[groupidx], workitems(ndrange)[idx])
end

Base.@propagate_inbounds function expand(ndrange::NDRange{N}, groupidx::CartesianIndex{N}, idx::Integer) where N
    expand(ndrange, groupidx, workitems(ndrange)[idx])
end

Base.@propagate_inbounds function expand(ndrange::NDRange{N}, groupidx::Integer, idx::CartesianIndex{N}) where N
    expand(ndrange, blocks(ndrange)[groupidx], idx)
end

"""
    partition(ndrange, workgroupsize)

Splits the maximum size of the iteration space by the workgroupsize.
Returns the number of workgroups necessary and whether the last workgroup
needs to perform dynamic bounds-checking.
"""
@inline function partition(ndrange, __workgroupsize)
    @assert length(__workgroupsize) <= length(ndrange)
    if length(__workgroupsize) < length(ndrange)
        # pad workgroupsize with ones
        workgroupsize = ntuple(Val(length(ndrange))) do I
            Base.@_inline_meta
            if I > length(__workgroupsize)
                return 1
            else
                return __workgroupsize[I]
            end
        end
    else
        workgroupsize = __workgroupsize
    end
    let workgroupsize = workgroupsize
        dynamic = Ref(false)
        blocks  = ntuple(Val(length(ndrange))) do I
            Base.@_inline_meta
            dynamic[] |= mod(ndrange[I], workgroupsize[I]) != 0
            return fld1(ndrange[I], workgroupsize[I])
        end
        return blocks, workgroupsize, dynamic[] ? DynamicCheck() : NoDynamicCheck()
    end
end

end #module
