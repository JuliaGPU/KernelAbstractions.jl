###
# Backend hierarchy
###


"""
    Backend

Abstract supertype for all KernelAbstractions backends.

Concrete backends (for example `CUDABackend` from CUDA.jl or `CPU` from KernelAbstractions)
determine where arrays are allocated and where kernels execute. Use [`get_backend`](@ref) to
obtain the backend for an array and [`allocate`](@ref) to create storage on a backend.

# Example

```julia
backend = get_backend(A)
kernel = my_kernel(backend, 256)
kernel(A, ndrange=length(A))
synchronize(backend)
```
"""
abstract type Backend end

"""
Abstract type for all GPU based KernelAbstractions backends.

!!! note
    New backend implementations **must** sub-type this abstract type.

!!! note
    `GPU` will be removed in KernelAbstractions v1.0
"""
abstract type GPU <: Backend end

"""
    get_backend(A::AbstractArray)::Backend

Get a [`Backend`](@ref) instance suitable for array `A`.

!!! note
    Backend implementations **must** provide `get_backend` for their custom array type.
    It should be the same as the return type of [`allocate`](@ref)
"""
function get_backend end

# Should cover SubArray, ReshapedArray, ReinterpretArray, Hermitian, AbstractTriangular, etc.:
function get_backend(A::AbstractArray)
    P = parent(A)
    if P isa typeof(A)
        throw(ArgumentError("Implement `KernelAbstractions.get_backend(::$(typeof(A)))`"))
    end
    return get_backend(P)
end
