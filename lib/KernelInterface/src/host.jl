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

Block the calling task until all work it has queued on the active device of `backend` has
completed.

!!! note
    Backend implementations **must** implement this function, and it **must** be
    cooperative: it may not block inside a driver call, but has to yield to the Julia
    scheduler while waiting. See the
    [notes for backend implementations](@ref implementations_notes) for why.
"""
function synchronize end

"""
    record_event(backend::Backend)

Capture the work the calling task has queued on `backend`'s currently active device so
far, and return a handle that [`wait_event`](@ref) can use to order later work after it,
either from another task or from the same task after switching devices.

The handle is only meaningful for the pair `record_event`/`wait_event`; do not use it for
anything else.

!!! note
    The default implementation calls [`synchronize`](@ref) and returns `nothing`.
    Backends whose queue is task-local **may** override this to return an event recorded
    on the current task's queue instead, without blocking the host. Such a backend
    **must** then also implement [`wait_event`](@ref) for the returned type. See the
    [notes for backend implementations](@ref implementations_notes).
"""
function record_event(backend::Backend)
    synchronize(backend)
    return nothing
end

"""
    wait_event(backend::Backend, event)

Order the work the calling task subsequently queues on `backend`'s currently active device
after the work captured by `event`, which was returned by [`record_event`](@ref).

The dependency is queue-ordered rather than task-ordered: it applies to the device that is
active when `wait_event` is called, and a later [`device!`](@ref) leaves the newly selected
device unordered with respect to `event`. Select the device first and wait afterwards:

```julia
event = record_event(backend)   # captures work on the current device
device!(backend, 2)
wait_event(backend, event)      # device 2 now waits for that work
```

!!! note
    `wait_event(::Backend, ::Nothing)` is a no-op, matching the default `record_event`.
    A backend that implements [`record_event`](@ref) **must** implement this for the event
    type it returns, either by enqueuing a dependency on the current task's queue, or by
    waiting cooperatively as [`synchronize`](@ref) does. A backend with more than one
    device **must** also accept an `event` that was recorded on a different device, by
    enqueuing the cross-device dependency if the driver supports one (CUDA's
    `cuStreamWaitEvent` does) and by waiting cooperatively otherwise. See the
    [notes for backend implementations](@ref implementations_notes).
"""
wait_event(::Backend, ::Nothing) = nothing

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
    Backends supporting multiple devices **must** implement `device(backend::Backend)::Int`,
    along with [`ndevices`](@ref), [`device!`](@ref) and `device(backend, A)`. The fallback
    only works for a single device, and throws if [`ndevices`](@ref) reports more.
"""
function device(backend::Backend)
    ndevices(backend) == 1 || throw_multi_device(device, backend)
    return 1
end

"""
    device(backend::Backend, A::AbstractArray)::Int

Return the 1-based index of the device that owns the memory of `A`, independently of the
currently active device.

!!! note
    Backends supporting multiple devices **must** implement this for their array type.
    The fallback only works for a single device, and throws if [`ndevices`](@ref) reports
    more.
"""
function device(backend::Backend, ::AbstractArray)
    ndevices(backend) == 1 || throw_multi_device(device, backend)
    return 1
end

"""
    ndevices(backend::Backend)::Int

Return the number of devices available to `backend`.

!!! note
    Backends supporting multiple devices **must** implement `ndevices(backend::Backend)::Int`,
    along with [`device`](@ref) and [`device!`](@ref). The fallback returns 1.
"""
function ndevices(::Backend)
    return 1
end

"""
    device!(backend::Backend, id::Int)::Nothing

Select the active device for `backend`. `id` is a 1-based device index; an `id` outside
`1:ndevices(backend)` throws an `ArgumentError`.

`device!` is not a synchronization point: work queued before the switch is not ordered
with respect to work queued after it. To order across a switch, either [`synchronize`](@ref)
beforehand, or bracket the switch with [`record_event`](@ref) and [`wait_event`](@ref).

# Example

```julia
device!(CUDABackend(), 2)  # use the second CUDA device
```

!!! note
    Backends supporting multiple devices **must** implement `device!(backend::Backend, id::Int)`,
    along with [`ndevices`](@ref) and [`device`](@ref). The fallback only works for a single
    device, and throws if [`ndevices`](@ref) reports more.
"""
function device!(backend::Backend, id::Int)
    n = ndevices(backend)
    if !(0 < id <= n)
        throw(ArgumentError("Device id $id out of bounds."))
    end
    n == 1 || throw_multi_device(device!, backend)
    return nothing
end

# a backend with several devices has to implement the device functions itself: the
# single-device fallbacks would silently answer for the wrong device
@noinline throw_multi_device(f, backend) =
    error("`$(typeof(backend))` has multiple devices, so it must implement `KernelInterface.$(nameof(f))`")

"""
    pagelock!(::Backend, dest::AbstractArray)::Union{Nothing, Missing}

Pagelock (pin) a host memory buffer for a backend device. This may be necessary for [`copyto!`](@ref)
to perform asynchronously with respect to the host.

This function returns `nothing`, or `missing` if not implemented.


!!! note
    Backends **may** implement this function.
"""
function pagelock!(::Backend, x)
    return missing
end

"""
    unsafe_free!(x::AbstractArray)

Release the memory of an array for reuse by future allocations, reducing pressure on the
allocator. The array may not be used afterwards.

This is a hint: releasing the memory is allowed to do nothing.

!!! note
    Backend implementations **may** implement this function for their array type, and
    should forward it to their own `unsafe_free!` if they have one. The fallback is a no-op.
"""
function unsafe_free! end

unsafe_free!(::AbstractArray) = return


"""
    supports_unified(::Backend)::Bool

Whether [`allocate`](@ref) supports `unified=true` on the active device: memory that can be
accessed from both the host and the device without explicit copies.

!!! note
    Backend implementations **must** implement this function if they support unified
    memory. The fallback returns `false`.
"""
supports_unified(::Backend) = false

"""
    supports_atomics(::Backend)::Bool

Whether kernels on the active device support Atomix.jl's atomic operations: at least `add`
and compare-and-swap on 32-bit integers and floats in global memory.

!!! note
    Backend implementations **must** implement this function if they support atomics.
    The fallback returns `false`.
"""
supports_atomics(::Backend) = false

"""
    supports_float64(::Backend)::Bool

Whether kernels on the active device support `Float64` values.

!!! note
    Backend implementations **must** implement this function if they support `Float64`.
    The fallback returns `false`.
"""
supports_float64(::Backend) = false

"""
    supports_subgroups(::Backend)::Bool

Whether kernels on the active device support sub-groups: the sub-group queries
([`get_sub_group_size`](@ref) etc.), [`sub_group_barrier`](@ref), and a fixed sub-group
width [`sub_group_size`](@ref). See the manual for what KernelInterface guarantees about
how work-groups are divided into sub-groups; a backend that can't ensure that reports
`false`.

Which types [`shfl_down`](@ref) supports is queried separately with [`supports_shuffle`](@ref).

!!! note
    Backend implementations **must** implement this function if they support sub-groups.
    The fallback returns `false`.
"""
supports_subgroups(::Backend) = false

"""
    supports_shuffle(::Backend, ::Type{T})::Bool

Whether kernels on the active device support [`shfl_down`](@ref) for values of type `T`.

!!! note
    Backend implementations **must** implement this function for the types they support.
    The fallback returns `false`.
"""
supports_shuffle(::Backend, ::Type) = false

"""
    allocate(::Backend, Type, dims...; unified=false)::AbstractArray

Allocate an uninitialized array on the active device of the backend. `unified=true`
allocates unified memory, accessible from the host and the device without explicit copies,
if the backend supports it and throws otherwise. Use [`supports_unified`](@ref) to
determine whether it is supported by a backend.

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

Allocate an array with [`allocate`](@ref) and fill it with zeros.

This is generic: backends implement `allocate` (and `fill!` for their array type).
"""
zeros(backend::Backend, T::Type, dims...; kwargs...) = zeros(backend, T, dims; kwargs...)
function zeros(backend::Backend, ::Type{T}, dims::Tuple; kwargs...) where {T}
    data = allocate(backend, T, dims; kwargs...)
    fill!(data, zero(T))
    return data
end

"""
    ones(::Backend, Type, dims...; unified=false)::AbstractArray

Allocate an array with [`allocate`](@ref) and fill it with ones.

This is generic: backends implement `allocate` (and `fill!` for their array type).
"""
ones(backend::Backend, T::Type, dims...; kwargs...) = ones(backend, T, dims; kwargs...)
function ones(backend::Backend, ::Type{T}, dims::Tuple; kwargs...) where {T}
    data = allocate(backend, T, dims; kwargs...)
    fill!(data, one(T))
    return data
end


"""
    copyto!(::Backend, dest::AbstractArray, src::AbstractArray)::typeof(dest)

Copy the elements of `src` to `dest`, ordered with respect to the other work on the calling
task's queue: after work queued before the copy, and before work queued after it. Returns
`dest`.

Either array can be a host array or an array of `backend`. `dest` and `src` must have the
same length, otherwise an `ArgumentError` is thrown. Backends only have to support dense
(contiguous) arrays with the same element type.

The copy may be asynchronous with respect to the host, but doesn't have to be: it can also
block until it has completed. For a simple, synchronous copy, use `Base.copyto!`.

!!! warning

    Because the copy may be asynchronous, the caller has to keep both arrays alive, and not
    access them from the host, until the copy has *completed*, e.g. by calling
    [`synchronize`](@ref) before using them. A `GC.@preserve` around `copyto!` only keeps
    them alive until the copy is queued:

    ```julia
    arr = zeros(64)
    GC.@preserve arr begin
        copyto!(backend, arr, ...)
        # other operations
        synchronize(backend)
    end
    ```

!!! note

    On some backends it may be necessary to first call [`pagelock!`](@ref) on host memory
    to enable fully asynchronous behavior w.r.t to the host.

!!! note
    Backends **must** implement this function, for host-to-device, device-to-host and
    device-to-device copies.
"""
function copyto! end
