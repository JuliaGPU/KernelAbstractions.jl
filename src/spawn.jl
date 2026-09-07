"""
    @spawn [threadpool] backend expr

Run `expr` on a new Julia task, like `Threads.@spawn`, while keeping the work queued on
`backend` correctly ordered between the two tasks. Returns the `Task`.

Backends may keep a separate queue (stream) per Julia task, so a kernel launched from one
task is not automatically ordered with respect to a kernel launched from another. `@spawn`
encodes the protocol that makes this safe:

1. Before the task starts, the work the spawning task has queued on `backend` is captured
   with [`record_event`](@ref KernelAbstractions.record_event). By default this is a full
   [`synchronize`](@ref); backends may instead record an event without blocking.
2. The new task selects the same device as the spawning task, then orders its own queue
   after the captured work with [`wait_event`](@ref KernelAbstractions.wait_event).
3. After `expr` returns, the task calls [`synchronize`](@ref) on `backend`, so that once
   `wait(task)` returns, all work the task queued has completed and its results may be used
   from any task. `fetch(task)` returns the value of `expr`.

The optional `threadpool` argument (`:default` or `:interactive`) is forwarded to
`Threads.@spawn`.

# Example

```julia
A = KernelAbstractions.ones(backend, Float32, 1024)
mul2_kernel(backend, 64)(A, ndrange = length(A))   # queued by the current task

task = KernelAbstractions.@spawn backend begin
    mul2_kernel(backend, 64)(A, ndrange = length(A))  # ordered after the launch above
    sum(A)
end
fetch(task) == 4 * length(A)
```

!!! note
    `expr` should not rely on data that the spawning task queues *after* `@spawn` returns.
    Order later work by waiting on the task, or by spawning again.

!!! note
    Steps 1 and 3 call [`synchronize`](@ref) on backends that have not opted into events.
    Backends should implement `synchronize` cooperatively, yielding to the Julia scheduler
    instead of blocking inside a driver call, so that spawned tasks can make progress
    concurrently. See [Notes for backend implementations](@ref implementations_notes).
"""
macro spawn(args...)
    if length(args) == 2
        threadpool = nothing
        backend, expr = args
    elseif length(args) == 3
        threadpool, backend, expr = args
    else
        throw(ArgumentError("@spawn expects `@spawn [threadpool] backend expr`"))
    end

    body = quote
        KI.device!(backend, dev)
        KI.wait_event(backend, event)
        local result = $(esc(expr))
        KI.synchronize(backend)
        result
    end
    task = if threadpool === nothing
        :(Threads.@spawn $body)
    else
        # `Threads.@spawn` inspects a literal `:default`/`:interactive`, so it must not be
        # escaped; anything else is an expression evaluated in the caller's scope.
        threadpool isa QuoteNode || (threadpool = esc(threadpool))
        :(Threads.@spawn $threadpool $body)
    end

    return quote
        local backend = $(esc(backend))
        local dev = KI.device(backend)
        local event = KI.record_event(backend)
        $task
    end
end
