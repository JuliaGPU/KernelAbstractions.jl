# operations that happen on the host side that don't belong in launch.jl

"""
    versioninfo(io::IO=stdout, backend::Backend)::Nothing

Print information about `backend` to `io`. It is up to the backends to
determine what is relevant.

!!! note
    Backend implementations **may** implement this function. If they do
    so, they should implement `versioninfo(io::IO, ::Backend)::Nothing`
"""
versioninfo(io::IO, b::Backend) = println(io, "`versioninfo` is not implemented for $b")
versioninfo(b::Backend) = versioninfo(stdout, b)

"""
    functional(::Backend)::Union{Bool, Missing}

Queries if the provided backend is functional. This may mean different
things for different backends, but generally should mean that the
necessary drivers and a compute device are available.

This function should return a `Bool` or `missing` if not implemented.

!!! compat "KernelAbstractions v0.9.22"
    This function was added in KernelAbstractions v0.9.22
"""
function functional(::Backend)
    return missing
end

"""
    synchronize(::Backend)

Synchronize the current backend.

!!! note
    Backend implementations **must** implement this function.
"""
function synchronize end

# Define:
#   adapt_storage(::Backend, a::Array) = adapt(BackendArray, a)
#   adapt_storage(::Backend, a::BackendArray) = a

"""
    priority!(::Backend, prio::Symbol)::Nothing

Set the priority for the backend stream/queue. This is an optional
feature that backends may or may not implement. If a backend shall
support priorities it must accept `:high`, `:normal`, `:low`.
Where `:normal` is the default.

!!! note
    Backend implementations **may** implement this function.
"""
function priority!(::Backend, prio::Symbol)
    if !(prio in (:high, :normal, :low))
        error("priority must be one of :high, :normal, :low")
    end
    return nothing
end

"""
    device(backend::Backend)::Int

Return the 1-based index of the currently active device for `backend`.

!!! note
    The default implementation assumes a single device. Backends supporting multiple devices
    **must** implement `device(backend::Backend)::Int`, [`ndevices`](@ref),
    and [`device!`](@ref).
"""
function device(::Backend)
    return 1
end

"""
    ndevices(backend::Backend)::Int

Return the number of devices available to `backend`.

!!! note
    The default implementation assumes a single device. Backends supporting multiple devices
    **must** implement `ndevices(backend::Backend)::Int`, [`device`](@ref),
    and [`device!`](@ref).
"""
function ndevices(::Backend)
    return 1
end

"""
    device!(backend::Backend, id::Int)::Nothing

Select the active device for `backend`. `id` is a 1-based device index and must satisfy
`1 <= id <= ndevices(backend)`.

# Example

```julia
device!(CUDABackend(), 2)  # use the second CUDA device
```

!!! note
    The default implementation assumes a single device. Backends supporting multiple devices
    **must** implement `device!(backend::Backend, id::Int)`, [`ndevices`](@ref),
    and [`device`](@ref).
"""
function device!(backend::Backend, id::Int)
    if !(0 < id <= ndevices(backend))
        throw(ArgumentError("Device id $id out of bounds."))
    end
    return nothing
end

"""
    pagelock!(::Backend, dest::AbstractArray)::Union{Nothing, Missing}

Pagelock (pin) a host memory buffer for a backend device. This may be necessary for [`copyto!`](@ref)
to perform asynchronously w.r.t to the host/

This function should return `nothing`; or `missing` if not implemented.


!!! note
    Backends **may** implement this function.
"""
function pagelock!(::Backend, x)
    return missing
end

"""
    unsafe_free!(x::AbstractArray)

Release the memory of an array for reuse by future allocations
and reduce pressure on the allocator.
After releasing the memory of an array, it should no longer be accessed.

!!! note
    On CPU backend this is always a no-op.

!!! note
    Backend implementations **may** implement this function.
    If not implemented for a particular backend, default action is a no-op.
    Otherwise, it should be defined for backend's array type.
"""
function unsafe_free! end

unsafe_free!(::AbstractArray) = return


"""
    supports_unified(::Backend)::Bool

Returns whether unified memory arrays are supported by the backend.

!!! note
    Backend implementations **should** implement this function
    only if they **do** support unified memory.
"""
supports_unified(::Backend) = false

"""
    supports_atomics(::Backend)::Bool

Returns whether `@atomic` operations are supported by the backend.

!!! note
    Backend implementations **must** implement this function
    only if they **do not** support atomic operations with Atomix.
"""
supports_atomics(::Backend) = true

"""
    supports_float64(::Backend)::Bool

Returns whether `Float64` values are supported by the backend.

!!! note
    Backend implementations **must** implement this function
    only if they **do not** support `Float64`.
"""
supports_float64(::Backend) = true

"""
    allocate(::Backend, Type, dims...; unified=false)::AbstractArray

Allocate a storage array appropriate for the computational backend. `unified=true`
allocates an array using unified memory if the backend supports it and throws otherwise.
Use [`supports_unified`](@ref) to determine whether it is supported by a backend.

!!! note
    Backend implementations **must** implement `allocate(::NewBackend, T, dims::Tuple)`
    Backend implementations **should** implement `allocate(::NewBackend, T, dims::Tuple; unified::Bool=false)`
"""
allocate(backend::Backend, T::Type, dims...; kwargs...) = allocate(backend, T, dims; kwargs...)
function allocate(backend::Backend, T::Type, dims::Tuple; unified::Union{Nothing, Bool} = nothing)
    if isnothing(unified)
        throw(MethodError(allocate, (backend, T, dims)))
    elseif unified
        throw(ArgumentError("`$(typeof(backend))` does not support unified memory. If you believe it does, please open a github issue."))
    else
        return allocate(backend, T, dims)
    end
end


"""
    zeros(::Backend, Type, dims...; unified=false)::AbstractArray

Allocate a storage array appropriate for the computational backend filled with zeros.
`unified=true` allocates an array using unified memory if the backend supports it and
throws otherwise.
"""
zeros(backend::Backend, T::Type, dims...; kwargs...) = zeros(backend, T, dims; kwargs...)
function zeros(backend::Backend, ::Type{T}, dims::Tuple; kwargs...) where {T}
    data = allocate(backend, T, dims...; kwargs...)
    fill!(data, zero(T))
    return data
end

"""
    ones(::Backend, Type, dims...; unified=false)::AbstractArray

Allocate a storage array appropriate for the computational backend filled with ones.
`unified=true` allocates an array using unified memory if the backend supports it and
throws otherwise.
"""
ones(backend::Backend, T::Type, dims...; kwargs...) = ones(backend, T, dims; kwargs...)
function ones(backend::Backend, ::Type{T}, dims::Tuple; kwargs...) where {T}
    data = allocate(backend, T, dims; kwargs...)
    fill!(data, one(T))
    return data
end


"""
    copyto!(::Backend, dest::AbstractArray, src::AbstractArray)

Perform an asynchronous `copyto!` operation that is execution ordered with respect to the back-end.

For most users, `Base.copyto!` should suffice, performance a simple, synchronous copy.
Only when you know you need asynchronicity w.r.t. the host, you should consider using
this asynchronous version, which requires additional lifetime guarantees as documented below.

!!! warning

    Because of the asynchronous nature of this operation, the user is required to guarantee that the lifetime
    of the source extends past the *completion* of the copy operation as to avoid a use-after-free. It is not
    sufficient to simply use `GC.@preserve` around the call to `copyto!`, because that only extends the
    lifetime past the operation getting queued. Instead, it may be required to `synchronize()`,
    or otherwise guarantee that the source will still be around when the copy is executed:

    ```julia
    arr = zeros(64)
    GC.@preserve arr begin
        copyto!(backend, arr, ...)
        # other operations
        synchronize(backend)
    end
    ```

!!! note

    On some back-ends it may be necessary to first call [`pagelock!`](@ref) on host memory
    to enable fully asynchronous behavior w.r.t to the host.

!!! note
    Backends **must** implement this function.
"""
function copyto! end
