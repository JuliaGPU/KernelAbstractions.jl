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

Backends should give each Julia task its own queue/stream, so that kernels
launched from different tasks can execute concurrently. This implies that work queued
from two tasks is not ordered with respect to each other.

[`KernelAbstractions.@spawn`](@ref) hides this from users by following a fixed protocol,
which backends can support with two optional functions:

- Before the new task is created, the spawning task calls
  [`record_event`](@ref KernelAbstractions.record_event) on the backend. The default
  implementation is a full [`synchronize`](@ref) returning `nothing`, which is always
  correct. A backend with task-local queues **may** instead record an event on the
  current task's queue and return it, so that the spawning task does not have to wait.
- The new task selects its device with [`device!`](@ref KernelAbstractions.device!) — the
  spawning task's, or the one the user asked for with `@spawn backend device=id` — and then
  calls [`wait_event`](@ref KernelAbstractions.wait_event) with the recorded handle. The
  order matters: `wait_event` makes the queue of the *currently active* device wait, so the
  device has to be selected first. A backend that overrides `record_event` **must**
  implement `wait_event` for its event type, typically by making the current task's queue
  wait on the event.
- After the user's code returns, the new task calls [`synchronize`](@ref), so that
  `wait(task)` in any other task implies that all work queued by the spawned task has
  completed.

A new Julia task does not inherit the device of the task that spawned it: backends keep the
active device in task-local state, which Julia does not copy into a child task, so the task
Backends with more than one device
**must** implement the device interface ([`device`](@ref KernelAbstractions.device),
[`ndevices`](@ref KernelAbstractions.ndevices), [`device!`](@ref KernelAbstractions.device!))
for `@spawn` to run on the right device.

`@spawn backend device=id` records the event on the spawning task's device but waits on
`id`, so a multi-device backend **must** accept an event recorded on a device other than the
one active in `wait_event`. A backend whose driver cannot **must** fall back
to waiting cooperatively, as [`synchronize`](@ref) does.

Because `device!` selects the queue that `wait_event` acts on, the same two functions are
what lets users order work across a device switch they make themselves:

```julia
event = KernelAbstractions.record_event(backend)
KernelAbstractions.device!(backend, 2)
KernelAbstractions.wait_event(backend, event)
```


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

## Iteration spaces and the validity of work items

The context a backend passes to a kernel carries the blocked iteration space,
[`__iterspace(ctx)`](@ref KernelAbstractions.NDIteration.NDRange), and the
`ndrange` of the launch, `__ndrange(ctx)`. A backend must derive everything
about a work item from these two objects through three functions:

- [`expand(iterspace, groupidx, idx)`](@ref KernelAbstractions.NDIteration.expand)
  gives the `CartesianIndex` handled by work item `idx` of workgroup `groupidx`;
- `expand(iterspace, groupidx, idx) in __ndrange(ctx)` tells whether that work
  item has an index to handle, which is how `__validindex` must be implemented;
- [`linear_index(__ndrange(ctx), I)`](@ref KernelAbstractions.NDIteration.linear_index)
  gives the linear index of `I`.

A backend must not assume that `__ndrange(ctx)` is a `CartesianIndices` or that
`expand` is an affine map: the `mapping` field of the `NDRange` lets a package
define its own iteration space, for example a list of indices to visit, by
extending these functions for its mapping type. Overriding `__validindex` or
`__index_Global_Linear` for a generic `ctx` would bypass such an extension.
See [`NDRange`](@ref KernelAbstractions.NDIteration.NDRange) for the functions
a custom mapping has to define.

