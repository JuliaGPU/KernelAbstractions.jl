"""
    @spawn [threadpool] backend expr

Run `expr` on a new Julia task, like `Threads.@spawn`, and return the `Task`. Use it in
place of `Threads.@spawn` to launch kernels from a task. It guarantees that

- the task runs on the device that was active in the spawning task;
- the work the task queues on `backend` runs after the work the spawning task had queued on
  `backend` before calling `@spawn`;
- once `wait(task)` or `fetch(task)` returns, all work the task queued on `backend` has
  completed, so its results may be used from any task. `fetch(task)` returns the value of
  `expr`.

Everything else works as for `Threads.@spawn`: the optional `threadpool` argument
(`:default` or `:interactive`) is forwarded, `\$x` captures the value of `x` at spawn time,
and an enclosing `@sync` waits for the task.

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

Backend authors: see the [notes for backend implementations](@ref implementations_notes)
for the protocol behind these guarantees, and for how to support it without a full
[`synchronize`](@ref).
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

    # The whole expansion is escaped so that `Threads.@spawn` is expanded in the caller's
    # scope: that is what lets an enclosing `@sync` see the task, and what makes `$x`
    # interpolation in `expr` work. Our own temporaries are gensyms so they cannot clash
    # with the user's variables.
    b, dev, event, result = gensym(:backend), gensym(:dev), gensym(:event), gensym(:result)
    body = quote
        $KI.device!($b, $dev)
        $KI.wait_event($b, $event)
        local $result = $expr
        $KI.synchronize($b)
        $result
    end
    task = if threadpool === nothing
        :(Threads.@spawn $body)
    else
        :(Threads.@spawn $threadpool $body)
    end

    return esc(
        quote
            local $b = $backend
            local $dev = $KI.device($b)
            local $event = $KI.record_event($b)
            $task
        end
    )
end
