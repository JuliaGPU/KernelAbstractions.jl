# Contiguous on-device arrays

export CLDeviceArray, CLDeviceVector, CLDeviceMatrix, CLLocalArray


## construction

# NOTE: we can't support the typical `tuple or series of integer` style construction,
#       because we're currently requiring a trailing pointer argument.

struct CLDeviceArray{T, N, A} <: DenseArray{T, N}
    ptr::LLVMPtr{T, A}
    maxsize::Int

    dims::Dims{N}
    len::Int

    # inner constructors, fully parameterized, exact types (ie. Int not <:Integer)
    # TODO: deprecate; put `ptr` first like oneArray
    CLDeviceArray{T, N, A}(
        dims::Dims{N}, ptr::LLVMPtr{T, A},
        maxsize::Int = prod(dims) * sizeof(T)
    ) where {T, A, N} =
        new(ptr, maxsize, dims, prod(dims))
end

const CLDeviceVector = CLDeviceArray{T, 1, A} where {T, A}
const CLDeviceMatrix = CLDeviceArray{T, 2, A} where {T, A}

# outer constructors, non-parameterized
CLDeviceArray(dims::NTuple{N, <:Integer}, p::LLVMPtr{T, A}) where {T, A, N} = CLDeviceArray{T, N, A}(dims, p)
CLDeviceArray(len::Integer, p::LLVMPtr{T, A}) where {T, A} = CLDeviceVector{T, A}((len,), p)

# outer constructors, partially parameterized
CLDeviceArray{T}(dims::NTuple{N, <:Integer}, p::LLVMPtr{T, A}) where {T, A, N} = CLDeviceArray{T, N, A}(dims, p)
CLDeviceArray{T}(len::Integer, p::LLVMPtr{T, A}) where {T, A} = CLDeviceVector{T, A}((len,), p)
CLDeviceArray{T, N}(dims::NTuple{N, <:Integer}, p::LLVMPtr{T, A}) where {T, A, N} = CLDeviceArray{T, N, A}(dims, p)
CLDeviceVector{T}(len::Integer, p::LLVMPtr{T, A}) where {T, A} = CLDeviceVector{T, A}((len,), p)

# outer constructors, fully parameterized
CLDeviceArray{T, N, A}(dims::NTuple{N, <:Integer}, p::LLVMPtr{T, A}) where {T, A, N} = CLDeviceArray{T, N, A}(Int.(dims), p)
CLDeviceVector{T, A}(len::Integer, p::LLVMPtr{T, A}) where {T, A} = CLDeviceVector{T, A}((Int(len),), p)


## array interface

Base.elsize(::Type{<:CLDeviceArray{T}}) where {T} = sizeof(T)

Base.size(g::CLDeviceArray) = g.dims
Base.sizeof(x::CLDeviceArray) = Base.elsize(x) * length(x)

# we store the array length too; computing prod(size) is expensive
Base.length(g::CLDeviceArray) = g.len

Base.pointer(x::CLDeviceArray{T, <:Any, A}) where {T, A} = Base.unsafe_convert(LLVMPtr{T, A}, x)
@inline function Base.pointer(x::CLDeviceArray{T, <:Any, A}, i::Integer) where {T, A}
    return Base.unsafe_convert(LLVMPtr{T, A}, x) + Base._memory_offset(x, i)
end

typetagdata(a::CLDeviceArray{<:Any, <:Any, A}, i = 1) where {A} =
    reinterpret(LLVMPtr{UInt8, A}, a.ptr + a.maxsize) + i - one(i)


## conversions

Base.unsafe_convert(::Type{LLVMPtr{T, A}}, x::CLDeviceArray{T, <:Any, A}) where {T, A} =
    x.ptr


## indexing intrinsics

# TODO: how are allocations aligned by the level zero API? keep track of this
#       because it enables optimizations like Load Store Vectorization
#       (cfr. shared memory and its wider-than-datatype alignment)

@generated function alignment(::CLDeviceArray{T}) where {T}
    return if Base.isbitsunion(T)
        _, sz, al = Base.uniontype_layout(T)
        al
    else
        Base.datatype_alignment(T)
    end
end

@device_function @inline function arrayref(A::CLDeviceArray{T}, index::Integer) where {T}
    @boundscheck checkbounds(A, index)
    if isbitstype(T)
        arrayref_bits(A, index)
    else #if isbitsunion(T)
        arrayref_union(A, index)
    end
end

@inline function arrayref_bits(A::CLDeviceArray{T}, index::Integer) where {T}
    align = alignment(A)
    return unsafe_load(pointer(A), index, Val(align))
end

@inline @generated function arrayref_union(A::CLDeviceArray{T, <:Any, AS}, index::Integer) where {T, AS}
    typs = Base.uniontypes(T)

    # generate code that conditionally loads a value based on the selector value.
    # lacking noreturn, we return T to avoid inference thinking this can return Nothing.
    ex = :(Base.llvmcall("unreachable", $T, Tuple{}))
    for (sel, typ) in Iterators.reverse(enumerate(typs))
        ex = quote
            if selector == $(sel - 1)
                ptr = reinterpret(LLVMPtr{$typ, AS}, data_ptr)
                unsafe_load(ptr, 1, Val(align))
            else
                $ex
            end
        end
    end

    return quote
        selector_ptr = typetagdata(A, index)
        selector = unsafe_load(selector_ptr)

        align = alignment(A)
        data_ptr = pointer(A, index)

        return $ex
    end
end

@device_function @inline function arrayset(A::CLDeviceArray{T}, x::T, index::Integer) where {T}
    @boundscheck checkbounds(A, index)
    if isbitstype(T)
        arrayset_bits(A, x, index)
    else #if isbitsunion(T)
        arrayset_union(A, x, index)
    end
    return A
end

@inline function arrayset_bits(A::CLDeviceArray{T}, x::T, index::Integer) where {T}
    align = alignment(A)
    return unsafe_store!(pointer(A), x, index, Val(align))
end

@inline @generated function arrayset_union(A::CLDeviceArray{T, <:Any, AS}, x::T, index::Integer) where {T, AS}
    typs = Base.uniontypes(T)
    sel = findfirst(isequal(x), typs)

    return quote
        selector_ptr = typetagdata(A, index)
        unsafe_store!(selector_ptr, $(UInt8(sel - 1)))

        align = alignment(A)
        data_ptr = pointer(A, index)

        unsafe_store!(reinterpret(LLVMPtr{$x, AS}, data_ptr), x, 1, Val(align))
        return
    end
end

@device_function @inline function const_arrayref(A::CLDeviceArray{T}, index::Integer) where {T}
    @boundscheck checkbounds(A, index)
    align = alignment(A)
    unsafe_cached_load(pointer(A), index, Val(align))
end


## indexing

Base.IndexStyle(::Type{<:CLDeviceArray}) = Base.IndexLinear()

Base.@propagate_inbounds Base.getindex(A::CLDeviceArray{T}, i1::Integer) where {T} =
    arrayref(A, i1)
Base.@propagate_inbounds Base.setindex!(A::CLDeviceArray{T}, x, i1::Integer) where {T} =
    arrayset(A, convert(T, x)::T, i1)

# preserve the specific integer type when indexing device arrays,
# to avoid extending 32-bit hardware indices to 64-bit.
Base.to_index(::CLDeviceArray, i::Integer) = i

# Base doesn't like Integer indices, so we need our own ND get and setindex! routines.
# See also: https://github.com/JuliaLang/julia/pull/42289
Base.@propagate_inbounds Base.getindex(
    A::CLDeviceArray,
    I::Union{Integer, CartesianIndex}...
) =
    A[Base._to_linear_index(A, to_indices(A, I)...)]
Base.@propagate_inbounds Base.setindex!(
    A::CLDeviceArray, x,
    I::Union{Integer, CartesianIndex}...
) =
    A[Base._to_linear_index(A, to_indices(A, I)...)] = x


## const indexing

"""
    Const(A::CLDeviceArray)

Mark a CLDeviceArray as constant/read-only. The invariant guaranteed is that you will not
modify an CLDeviceArray for the duration of the current kernel.

This API can only be used on devices with compute capability 3.5 or higher.

!!! warning
    Experimental API. Subject to change without deprecation.
"""
struct Const{T, N, AS} <: DenseArray{T, N}
    a::CLDeviceArray{T, N, AS}
end
Base.Experimental.Const(A::CLDeviceArray) = Const(A)

Base.IndexStyle(::Type{<:Const}) = IndexLinear()
Base.size(C::Const) = size(C.a)
Base.axes(C::Const) = axes(C.a)
Base.@propagate_inbounds Base.getindex(A::Const, i1::Integer) = const_arrayref(A.a, i1)


## other

Base.show(io::IO, a::CLDeviceVector) =
    print(io, "$(length(a))-element device array at $(pointer(a))")
Base.show(io::IO, a::CLDeviceArray) =
    print(io, "$(join(a.shape, '×')) device array at $(pointer(a))")

Base.show(io::IO, mime::MIME"text/plain", a::CLDeviceArray) = show(io, a)

@inline function Base.iterate(A::CLDeviceArray, i = 1)
    return if (i % UInt) - 1 < length(A)
        (@inbounds A[i], i + 1)
    else
        nothing
    end
end

function Base.reinterpret(::Type{T}, a::CLDeviceArray{S, N, A}) where {T, S, N, A}
    err = _reinterpret_exception(T, a)
    err === nothing || throw(err)

    if sizeof(T) == sizeof(S) # fast case
        return CLDeviceArray{T, N, A}(size(a), reinterpret(LLVMPtr{T, A}, a.ptr), a.maxsize)
    end

    isize = size(a)
    size1 = div(isize[1] * sizeof(S), sizeof(T))
    osize = tuple(size1, Base.tail(isize)...)
    return CLDeviceArray{T, N, A}(osize, reinterpret(LLVMPtr{T, A}, a.ptr), a.maxsize)
end


## local memory

# XXX: use OpenCL-style local memory arguments instead?

@inline function CLLocalArray(::Type{T}, dims) where {T}
    len = prod(dims)
    # NOTE: this relies on const-prop to forward the literal length to the generator.
    #       maybe we should include the size in the type, like StaticArrays does?
    ptr = emit_localmemory(T, Val(len))
    return CLDeviceArray(dims, ptr)
end
