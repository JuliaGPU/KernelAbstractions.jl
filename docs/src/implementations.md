# [Notes for backend implementations](@id implementations_notes)

The [KernelInterface](@ref kernelinterface) sibling package defines the core interface a backend must implement. A backend must implement a backend type that subtypes `KernelInterface.Backend`. This documentation contains the host and devices side functions that backends can define, as well as whether they are mandatory or not.

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


## Launching `@kernel` kernels

A kernel written with [`@kernel`](@ref) receives a hidden context, a
`KernelAbstractions.CompilerMetadata` built by the backend's `mkcontext`, from which
[`@index`](@ref) computes its indices. By default (a context without a `launch`),
`@index` assumes that the kernel was launched on a 1-D grid of
`length(blocks(iterspace))` groups of `length(workitems(iterspace))` work-items. It then
decomposes the linear hardware ids into Cartesian positions, which takes integer divisions
when the `ndrange` is not known at compile time, and computes in `Int`.

A backend **may** launch kernels differently, and pass the `launch` keyword to the
`CompilerMetadata` constructor to tell `@index` how:

- [`NDLaunch{T}`](@ref KernelAbstractions.NDLaunch): the grid has the shape of the
  iteration space (for as many dimensions as the backend's grid has, i.e. up to 3), so
  `@index` doesn't need any divisions;
- [`LinearLaunch{T}`](@ref KernelAbstractions.LinearLaunch): the default 1-D grid.

Either way `@index` computes in `T`, e.g. `Int32`, which is faster on GPUs. The backend
has to implement the typed [`KI.get_group_id`](@ref KernelInterface.get_group_id) and
[`KI.get_local_id`](@ref KernelInterface.get_local_id) queries such that they compute in
`T` too, e.g. without checked conversions.

[`select_launch`](@ref KernelAbstractions.select_launch) chooses the launch from the
iteration space, whether the workgroup size will be tuned, and the limits of the backend
([`KI.max_work_group_size`](@ref KernelInterface.max_work_group_size),
[`KI.max_work_group_dims`](@ref KernelInterface.max_work_group_dims) and
[`KI.max_num_groups`](@ref KernelInterface.max_num_groups)). It doesn't depend on the
workgroup size a backend tunes afterwards, which keeps the context type (and thus the
compiled kernel) the same before and after tuning, as long as the backend tunes with
[`launch_workgroupsize`](@ref KernelAbstractions.launch_workgroupsize). A launch then
looks like this:

```julia
function (obj::KA.Kernel{MyBackend})(args...; ndrange = nothing, workgroupsize = nothing)
    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, workgroupsize)
    launch = KA.select_launch(obj, workgroupsize, iterspace)
    ctx = KA.CompilerMetadata{KA.ndrange(obj), KA.DynamicCheck}(ndrange, iterspace; launch)
    kernel = compile(obj.f, ctx, args...)

    if KA.workgroupsize(obj) <: KA.DynamicSize && workgroupsize === nothing
        threads = max_threads(kernel)  # at most `KI.max_work_group_size(backend)`
        workgroupsize = KA.launch_workgroupsize(backend, launch, threads, ndrange)
        iterspace, dynamic = KA.partition(obj, ndrange, workgroupsize)
        ctx = KA.CompilerMetadata{KA.ndrange(obj), KA.DynamicCheck}(ndrange, iterspace; launch)
    end

    groups, items = size(KA.blocks(iterspace)), size(KA.workitems(iterspace))
    prod(groups) == 0 && return
    if launch isa KA.NDLaunch
        run(kernel, ctx, args...; groups, items)            # padded to 3 dimensions
    else
        run(kernel, ctx, args...; groups = prod(groups), items = prod(items))
    end
end
```

The POCL backend is an example. Backends that launch on an N-d grid **must not** override
`__validindex` or the `__index_*` functions, which dispatch on the launch.

Packages that customize the iteration space (with a custom `partition` and `expand`)
don't need to do anything for these launches: the index functions only compute the global
index directly for the iteration spaces KernelAbstractions creates itself, and call
`expand`, `in` and `linear_index` otherwise.

Packages with an `Adapt` rule for `CompilerMetadata` **must** preserve its `launch`, e.g. by
passing `launch = KernelAbstractions.__launch(ctx)` to the constructor. Otherwise the
kernel computes its indices as if it had been launched on a 1-D grid, which gives wrong
results for a kernel launched with an `NDLaunch`.
