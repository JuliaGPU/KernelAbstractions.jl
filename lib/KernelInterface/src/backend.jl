###
# Backend hierarchy
###


"""
    Backend

Abstract supertype for all KernelInterface backends. Backends subtype it directly.

A backend value identifies a backend and its configuration (e.g. compiler options). The
device and the queue that operations use are task-local: each task selects its active device
with [`device!`](@ref). Host-side queries answer for the calling task's active device, and
work is queued on the calling task's queue of that device. Use [`get_backend`](@ref) to obtain the
backend of an array and [`allocate`](@ref) to create storage on a backend.

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
