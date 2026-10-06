# Per-work-item private memory, as returned by `@private`.

import GPUCompiler
using LLVM.Interop: LLVMPtr
using StaticArraysCore: StaticArray, size_to_tuple

"""
    PrivateArray{T,S,N,L} <: StaticArray{S,T,N}

Fixed-size array in per-work-item stack storage, as returned by [`@private`](@ref). The
storage is uninitialized, lives until the kernel returns, and is shared by all copies of the
array object. Like a local array in C, every `@private` declaration has a single allocation:
arrays created by the same declaration, e.g. in different iterations of a loop, share storage.

The type only implements indexing; everything else uses the generic `AbstractArray`
implementations, or StaticArrays' if that package is loaded. See [`@private`](@ref) for what
that means in a kernel. It cannot be constructed from values, so `copy`, `zero` and other
methods that construct a new array of the same type are not supported.
"""
struct PrivateArray{T, S <: Tuple, N, L} <: StaticArray{S, T, N}
    # GPUCompiler places the alloca in the target's alloca address space and casts it to the
    # requested one. LLVM's default address space is valid everywhere: private memory on
    # SPIR-V and Metal, and a generic pointer on NVPTX and AMDGPU, which LLVM infers back to
    # private memory. A `Ptr` would not do: it is an integer in LLVM IR before Julia 1.12, which
    # keeps the alloca from being promoted, and its loads are not aligned.
    ptr::LLVMPtr{T, 0}

    # only from a pointer: constructing one from values would need an allocation that
    # outlives the constructor
    PrivateArray{T, S, N, L}(ptr::LLVMPtr{T, 0}) where {T, S, N, L} = new{T, S, N, L}(ptr)
end

# Allocates the storage for `@private`. It takes the kernel context so that it can only be used
# in a kernel: the alloca lives in the function that contains it, so a helper must not create
# one and return it. It is only defined for device code, as host code calling it would contain
# an alloca that can't be lowered.
function Scratchpad end
Base.Experimental.@overlay GPUCompiler.SHARED_METHOD_TABLE @inline function Scratchpad(
        ctx, ::Type{T}, ::Val{Dims}
    ) where {T, Dims}
    L = prod(Dims)
    ptr = GPUCompiler.alloca(T, Val(L), Val(0))
    return PrivateArray{T, Tuple{Dims...}, length(Dims), L}(ptr)
end

Base.size(::PrivateArray{T, S}) where {T, S} = size_to_tuple(S)
Base.length(::PrivateArray{T, S, N, L}) where {T, S, N, L} = L
Base.IndexStyle(::Type{<:PrivateArray}) = IndexLinear()

@inline function Base.getindex(p::PrivateArray{T}, i::Int) where {T}
    @boundscheck checkbounds(p, i)
    return unsafe_load(p.ptr, i, Val(Base.datatype_alignment(T)))
end

@inline function Base.setindex!(p::PrivateArray{T}, x, i::Int) where {T}
    @boundscheck checkbounds(p, i)
    unsafe_store!(p.ptr, convert(T, x), i, Val(Base.datatype_alignment(T)))
    return p
end
