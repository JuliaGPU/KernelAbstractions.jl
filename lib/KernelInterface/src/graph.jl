# Graph capture: record the device work a function enqueues once, and replay it with less
# host overhead. Backends without graphs run the function eagerly instead, which is
# indistinguishable as long as the function follows the contract documented on `capture`.

"""
    GraphCaptureError(msg)

Thrown by [`capture`](@ref) and [`update!`](@ref) with `strict = true` when a backend that
supports graphs failed to record the function.
"""
struct GraphCaptureError <: Exception
    msg::String
end
Base.showerror(io::IO, err::GraphCaptureError) = print(io, "GraphCaptureError: ", err.msg)

"""
    supports_graphs(::Backend)::Bool

Whether the backend can record the device work of a function into a graph and replay it
(see [`capture`](@ref)). Without graphs, [`capture`](@ref) still works, but replaying a
[`Graph`](@ref) runs the function again.

!!! note
    Backend implementations **must** implement this function if they support graphs,
    together with [`capture_graph`](@ref) and [`launch_graph`](@ref). The fallback
    returns `false`.
"""
supports_graphs(::Backend) = false

"""
    capture_graph(backend::Backend, f)

Record the device work that calling `f()` enqueues on the calling task's queue of the
active device of `backend`, without executing that work, and return a handle for
[`launch_graph`](@ref). Return `nothing` if recording failed, e.g. because `f` performed an
operation that cannot be recorded; the caller then runs `f` eagerly instead.

The handle must keep the memory that the recorded operations use alive, or rely on the
[`Graph`](@ref) holding on to `f`, which keeps the arrays captured by `f` alive.

!!! note
    Backends that support graphs **must** implement this function. The fallback returns
    `nothing`. It is only called if [`supports_graphs`](@ref) returns `true`, and it must
    not leave the backend in a capturing state when `f` throws.
"""
capture_graph(::Backend, f) = nothing

"""
    launch_graph(backend::Backend, handle)

Enqueue the work recorded in `handle`, returned by [`capture_graph`](@ref) or
[`update_graph`](@ref), on the calling task's queue of the active device of `backend`.
Like a kernel launch, this does not wait for the work to complete: [`synchronize`](@ref)
does, and [`record_event`](@ref)/[`wait_event`](@ref) order other work around it.

!!! note
    Backends that support graphs **must** implement this function for their handle type.
"""
function launch_graph end

"""
    update_graph(backend::Backend, handle, f)

Record `f` again, like [`capture_graph`](@ref), and return a handle that replays the new
recording: `handle` updated in place where possible (e.g. when only kernel arguments
changed), or a new handle otherwise. Return `nothing` if recording failed.

`handle` is not used again by the caller, so the backend may release it if it returns a
new one.

!!! note
    Backends **may** implement this function, typically to update `handle` in place, which
    is cheaper than instantiating a new graph. The fallback calls
    [`capture_graph`](@ref)`(backend, f)`.
"""
update_graph(backend::Backend, handle, f) = capture_graph(backend, f)

"""
    Graph

The device work of a function, recorded by [`capture`](@ref). Call it, `graph()`, to
enqueue that work again. Re-record it with [`update!`](@ref). Whether the work was
actually recorded, rather than being executed eagerly on every call, is reported by
[`is_captured`](@ref).

A `Graph` keeps the function it was recorded from, and with it the arrays that function
captures, alive. Calls of the same `Graph` are serialized.
"""
mutable struct Graph{B <: Backend}
    const backend::B
    f::Any
    # the backend's handle, or `nothing` to run `f` eagerly
    handle::Any
    const lock::ReentrantLock
end

function Base.show(io::IO, graph::Graph)
    print(io, "Graph(", graph.backend, is_captured(graph) ? ", captured)" : ", eager)")
    return
end

"""
    is_captured(graph::Graph)::Bool

Whether `graph` replays recorded device work, rather than running its function eagerly
because the backend doesn't [support graphs](@ref supports_graphs) or recording failed.
"""
is_captured(graph::Graph) = graph.handle !== nothing

# record `f`, either from scratch or by updating `handle`; `nothing` falls back to eager
function record(backend::Backend, handle, f, strict::Bool)
    supports_graphs(backend) || return nothing
    new_handle = handle === nothing ? capture_graph(backend, f) : update_graph(backend, handle, f)
    if new_handle === nothing && strict
        throw(GraphCaptureError("`$(typeof(backend))` failed to record the function into a graph"))
    end
    return new_handle
end

"""
    capture(f, backend::Backend; warmup = true, strict = false)::Graph

Record the device work that `f()` enqueues on `backend` into a [`Graph`](@ref), which can
then be replayed many times, with much less host overhead than performing the operations
one by one:

```julia
graph = capture(backend) do
    kernel1(backend)(y, x; ndrange = length(y))
    kernel2(backend)(z, y; ndrange = length(z))
end
for i in 1:100
    graph()   # enqueues both kernels
end
synchronize(backend)
```

Graphs are an optional feature: on backends that don't [support graphs](@ref
supports_graphs), or if recording fails, calling the graph calls `f()`. Code that follows
the contract below gives the same results either way.

# Contract

The graph replays the operations with the arguments they were recorded with. So `f` must
only enqueue device work on `backend`, from the calling task, on the active device:

- kernel launches, device-to-device copies, and fills or broadcasts into existing arrays;
- no [`synchronize`](@ref), no scalar indexing of device arrays, and no copies between the
  host and the device;
- no allocations, including temporaries, e.g. from broadcasting into a new array;
- no other tasks, queues or devices, e.g. no [`device!`](@ref) and no
  `KernelAbstractions.@spawn`.

Between calls of the graph, only the contents of the arrays used by `f` may change. To
change which arrays, kernels, launch sizes or scalar arguments `f` uses, record it again
with [`update!`](@ref), or use [`@captured`](@ref). Keep scalars that change from call to
call in a device array instead.

# Keywords

- `warmup`: call `f()` once before recording, so that kernels are compiled and libraries
  initialized, which recording may not be able to do. This executes the work of `f` once.
  With `warmup = false`, `capture` only records, without executing anything.
- `strict`: throw a [`GraphCaptureError`](@ref) instead of falling back to eager execution
  when a backend that supports graphs fails to record `f`.

The graph has to be called with the device active that it was recorded on.
"""
function capture(f, backend::Backend; warmup::Bool = true, strict::Bool = false)
    warmup && f()
    handle = record(backend, nothing, f, strict)
    return Graph{typeof(backend)}(backend, f, handle, ReentrantLock())
end

function (graph::Graph)()
    lock(graph.lock) do
        handle = graph.handle
        if handle === nothing
            graph.f()
        else
            launch_graph(graph.backend, handle)
        end
    end
    return nothing
end

"""
    update!(graph::Graph, f = graph.f; strict = false)::Graph

Record `f` again into `graph`, without executing its work, e.g. because it uses different
arrays or scalar arguments than when `graph` was recorded. Backends update the recorded
work in place where possible, which is much cheaper than recording a new graph.

If recording fails, `graph` falls back to running `f` eagerly, or a
[`GraphCaptureError`](@ref) is thrown if `strict = true`.
"""
function update!(graph::Graph, f = graph.f; strict::Bool = false)
    lock(graph.lock) do
        graph.handle = record(graph.backend, graph.handle, f, strict)
        graph.f = f
    end
    return graph
end

# the graphs of a `@captured` call site, one per task, so that tasks don't serialize on
# each other or replay each other's recording
struct CapturedGraphs
    graphs::WeakKeyDict{Task, Graph}
end
CapturedGraphs() = CapturedGraphs(WeakKeyDict{Task, Graph}())

"""
    @captured backend expr

Enqueue the device work of `expr` on `backend` through a graph, recorded the first time
the call site is reached by a task, and updated on later calls. Updating is cheap if
`expr` performs the same operations, even with different arrays or scalar arguments, so
`expr` does not need to follow the restrictions on reusing a [`Graph`](@ref).

```julia
for t in times
    @captured backend begin
        step1(backend)(u, t; ndrange = length(u))
        step2(backend)(u; ndrange = length(u))
    end
end
```

Since `expr` itself still runs every time, this saves less host overhead than calling a
[`Graph`](@ref) recorded with [`capture`](@ref), but it still removes the gaps between short
kernels on the device. The other restrictions of `capture` apply to
`expr`: e.g., it must not synchronize or allocate. On backends that don't
[support graphs](@ref supports_graphs), `expr` is executed normally.
"""
macro captured(backend, ex)
    cache = CapturedGraphs()
    return quote
        captured($(esc(backend)), $cache) do
            $(esc(ex))
        end
    end
end

function captured(f, backend::Backend, cache::CapturedGraphs)
    if !supports_graphs(backend)
        f()
        return nothing
    end
    task = current_task()
    graph = get(cache.graphs, task, nothing)
    if graph === nothing || graph.backend != backend
        # the warm-up executes the work of `f`, so there is nothing to launch this time
        cache.graphs[task] = capture(f, backend)
        return nothing
    end
    update!(graph, f)
    graph()
    return nothing
end
