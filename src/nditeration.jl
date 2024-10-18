module NDIteration

import Base.MultiplicativeInverses: SignedMultiplicativeInverse

# CartesianIndex uses Int instead of Int32

@eval EmptySMI() = $(Expr(:new, SignedMultiplicativeInverse{Int32}, Int32(0), typemax(Int32), 0 % Int8, 0 % UInt8))
SMI(i) = i == 0 ? EmptySMI() : SignedMultiplicativeInverse{Int32}(i)

struct FastCartesianIndices{N} <: AbstractArray{CartesianIndex{N}, N}
    inverses::NTuple{N, SignedMultiplicativeInverse{Int32}}
end

function FastCartesianIndices(indices::NTuple{N}) where {N}
    inverses = map(i -> SMI(Int32(i)), indices)
    FastCartesianIndices(inverses)
end

function Base.size(FCI::FastCartesianIndices{N}) where {N}
    ntuple(Val(N)) do I
        FCI.inverses[I].divisor
    end
end

@inline function Base.getindex(::FastCartesianIndices{0})
    return CartesianIndex()
end

@inline function Base.getindex(iter::FastCartesianIndices{N}, I::Vararg{Int, N}) where {N}
    @boundscheck checkbounds(iter, I...)
    index = map(iter.inverses, I) do inv, i
        @inbounds getindex(Base.OneTo(inv.divisor), i%Int32)
    end
    CartesianIndex(index)
end

_ind2sub_recuse(::Tuple{}, ind) = (ind + 1,)
function _ind2sub_recurse(indslast::NTuple{1}, ind)
    Base.@_inline_meta
    (_lookup(ind, indslast[1]),)
end

function _ind2sub_recurse(inds, ind)
    Base.@_inline_meta
    assume(ind >= 0)
    inv = inds[1]
    indnext, f, l = _div(ind, inv)
    (ind - l * indnext + f, _ind2sub_recurse(Base.tail(inds), indnext)...)
end

_lookup(ind, inv::SignedMultiplicativeInverse) = ind + 1
function _div(ind, inv::SignedMultiplicativeInverse)
    # inv.divisor == 0 && throw(DivideError())
    assume(ind >= 0)
    div(ind % Int32, inv), 1, inv.divisor
end

function Base._ind2sub(inv::FastCartesianIndices, ind)
    Base.@_inline_meta
    _ind2sub_recurse(inv.inverses, ind - 1)
end

export _Size, StaticSize, DynamicSize, get
export NDRange, blocks, workitems, expand
export DynamicCheck, NoDynamicCheck

import Base.@pure

struct DynamicCheck end
struct NoDynamicCheck end

abstract type _Size end
struct DynamicSize <: _Size end
struct StaticSize{S} <: _Size
    function StaticSize{S}() where {S}
        new{S::Tuple{Vararg{Int}}}()
    end
end

@pure StaticSize(s::Tuple{Vararg{Int}}) = StaticSize{s}()
@pure StaticSize(s::Int...) = StaticSize{s}()
@pure StaticSize(s::Type{<:Tuple}) = StaticSize{tuple(s.parameters...)}()

# Some @pure convenience functions for `StaticSize`
@pure get(::Type{StaticSize{S}}) where {S} = S
@pure get(::StaticSize{S}) where {S} = S
@pure Base.getindex(::StaticSize{S}, i::Int) where {S} = i <= length(S) ? S[i] : 1
@pure Base.ndims(::StaticSize{S}) where {S} = length(S)
@pure Base.length(::StaticSize{S}) where {S} = prod(S)


"""
    NDRange

Encodes a blocked iteration space. 

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
struct NDRange{N, StaticBlocks, StaticWorkitems, DynamicBlock, DynamicWorkitems}
    blocks::DynamicBlock
    workitems::DynamicWorkitems

    function NDRange{N, B, W}(blocks::Union{Nothing, FastCartesianIndices{N}}, workitems::Union{Nothing, FastCartesianIndices{N}}) where {N, B, W}
        @assert B <: _Size
        @assert W <: _Size
        new{N, B, W, typeof(blocks), typeof(workitems)}(blocks, workitems)
    end
end

function NDRange{N, B, W}() where {N, B, W}
    NDRange{N, B, W}(nothing, nothing)
end

function NDRange{N, B, W}(blocks::CartesianIndices, workitems::CartesianIndices) where {N, B, W}
    return NDRange{N, B, W}(FastCartesianIndices(size(blocks)), FastCartesianIndices(size(workitems)))
end

function NDRange{N, B, W}(blocks::Nothing, workitems::CartesianIndices) where {N, B, W}
    return NDRange{N, B, W}(blocks, FastCartesianIndices(size(workitems)))
end

function NDRange{N, B, W}(blocks::CartesianIndices, workitems::Nothing) where {N, B, W}
    return NDRange{N, B, W}(FastCartesianIndices(size(blocks)), workitems)
end

@inline workitems(range::NDRange{N, B, W}) where {N, B, W <: DynamicSize} = range.workitems::FastCartesianIndices{N}
@inline workitems(range::NDRange{N, B, W}) where {N, B, W <: StaticSize} = CartesianIndices(get(W))::CartesianIndices{N}
@inline blocks(range::NDRange{N, B}) where {N, B <: DynamicSize} = range.blocks::FastCartesianIndices{N}
@inline blocks(range::NDRange{N, B}) where {N, B <: StaticSize} = CartesianIndices(get(B))::CartesianIndices{N}

import Base.iterate
@inline iterate(range::NDRange) = iterate(blocks(range))
@inline iterate(range::NDRange, state) = iterate(blocks(range), state)

Base.length(range::NDRange) = length(blocks(range))

@inline function expand(ndrange::NDRange{N}, groupidx::CartesianIndex{N}, idx::CartesianIndex{N}) where {N}
    nI = ntuple(Val(N)) do I
        Base.@_inline_meta
        stride = size(workitems(ndrange), I)
        gidx = groupidx.I[I]
        (gidx - 1) * stride + idx.I[I]
    end
    CartesianIndex(nI)
end

Base.@propagate_inbounds function expand(ndrange::NDRange{N}, groupidx::Integer, idx::Integer) where {N}
    return expand(ndrange, blocks(ndrange)[groupidx], workitems(ndrange)[idx])
end

Base.@propagate_inbounds function expand(ndrange::NDRange{N}, groupidx::CartesianIndex{N}, idx::Integer) where {N}
    expand(ndrange, groupidx, workitems(ndrange)[idx])
end

Base.@propagate_inbounds function expand(ndrange::NDRange{N}, groupidx::Integer, idx::CartesianIndex{N}) where {N}
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
        blocks = ntuple(Val(length(ndrange))) do I
            Base.@_inline_meta
            dynamic[] |= mod(ndrange[I], workgroupsize[I]) != 0
            return fld1(ndrange[I], workgroupsize[I])
        end

        return blocks, workgroupsize, dynamic[] ? DynamicCheck() : NoDynamicCheck()
    end
end



"""
    assume(cond::Bool)
Assume that the condition `cond` is true. This is a hint to the compiler, possibly enabling
it to optimize more aggressively.
"""
@inline assume(cond::Bool) = Base.llvmcall(("""
        declare void @llvm.assume(i1)
        define void @entry(i8) #0 {
            %cond = icmp eq i8 %0, 1
            call void @llvm.assume(i1 %cond)
            ret void
        }
        attributes #0 = { alwaysinline }""", "entry"),
    Nothing, Tuple{Bool}, cond)
end #module
