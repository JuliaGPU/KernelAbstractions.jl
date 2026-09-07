# [Notes for backend implementations](@id implementations_notes)

The [KernelInterface](@ref kernelinterface) sibling package defines the core interface a backend must implement. A backend must implement a backend type that subtypes `KernelInterface.GPU`, or `KernelInterface.Backend` for non-gpu backends. This documentation contains the host and devices side functions that backends can define, as well as whether they are mandatory or not.

## Semantics of `KernelAbstractions.synchronize`

[`KernelAbstractions.synchronize`](@ref) is required to be **cooperative**,
with that we mean it can not block inside an external library, but instead must
implement a cooperative wait that will `yield` the current task and return the
scheduling slice to the Julia runtime.

This is of particular import to allow for overlapping of communication and
computation with MPI, and for [`KernelAbstractions.@spawn`](@ref), whose
trailing `synchronize` would otherwise stall every task scheduled on the same
thread instead of letting independent tasks run concurrently.

## Task-local queues and `KernelAbstractions.@spawn`

Backends are free to give each Julia task its own queue (stream), so that kernels
launched from different tasks can execute concurrently. The price is that work queued
from two tasks is not ordered with respect to each other, and that a task waiting on
another task with `wait` learns nothing about the state of that task's queue.

[`KernelAbstractions.@spawn`](@ref) hides this from users by following a fixed protocol,
which backends can support with two optional functions:

- Before the new task is created, the spawning task calls
  [`record_event`](@ref KernelAbstractions.record_event) on the backend. The default
  implementation is a full [`synchronize`](@ref) returning `nothing`, which is always
  correct. A backend with task-local queues **may** instead record an event on the
  current task's queue and return it, so that the spawning task does not have to wait.
- The new task first selects the spawning task's device with [`device!`](@ref KernelAbstractions.device!),
  then calls [`wait_event`](@ref KernelAbstractions.wait_event) with the recorded handle.
  A backend that overrides `record_event` **must** implement `wait_event` for its event
  type, typically by making the current task's queue wait on the event.
- After the user's code returns, the new task calls [`synchronize`](@ref), so that
  `wait(task)` in any other task implies that all work queued by the spawned task has
  completed.

A backend with a single, global queue needs no changes: the defaults are exactly the
"synchronize before, synchronize after" discipline users would otherwise write by hand.


## Moving data with `adapt`

KernelAbstractions extends [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl) so
that [`adapt(backend, x)`](@ref Adapt.adapt_storage(::Backend, ::Any)) moves the
arrays in `x` to `backend`, without the caller having to know the backend's
array type. Every backend **must** support this by extending
`Adapt.adapt_storage` for its backend type. The recommended definition delegates
to the backend's array type, so that `adapt(backend, x)` and
`adapt(BackendArray, x)` agree:

```julia
Adapt.adapt_storage(::CUDABackend, x) = adapt(CuArray, x)
```

