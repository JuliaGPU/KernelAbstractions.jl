"""
    @spawn [threadpool] backend [device=id] expr

Run `expr` on a new Julia task, like `Threads.@spawn`, and return the `Task`. Use it in
place of `Threads.@spawn` to launch kernels from a task. It guarantees that

- the task runs on the device that was active in the spawning task, or on `device` when
  that argument is given;
- the work the task queues on `backend` runs after the work the spawning task had queued on
  `backend` before calling `@spawn`;
- once `wait(task)` or `fetch(task)` returns, all work the task queued on `backend` has
  completed, so its results may be used from any task. `fetch(task)` returns the value of
  `expr`. If `expr` throws, the task is not synchronized: its queued work may still be
  running when the exception surfaces.

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

# Choosing the device

Backends keep the active device in task-local state, and Julia does not copy that state
into a child task. A task started with plain `Threads.@spawn` therefore runs on the
backend's *default* device, whichever device the spawning task was using. `@spawn` selects
the device explicitly instead: by default the one active in the spawning task, or the one
named by `device`, a 1-based index into `1:ndevices(backend)`:

```julia
task = KernelAbstractions.@spawn backend device=2 begin
    mul2_kernel(backend, 64)(B, ndrange = length(B))
end
```

The ordering guarantee holds across that switch: the task's work on `device` is still
ordered after the work the spawning task had queued on *its* device. Backends that support
more than one device implement this with a cross-device
[`wait_event`](@ref KernelAbstractions.wait_event).

!!! note
    `expr` should not rely on data that the spawning task queues *after* `@spawn` returns.
    Order later work by waiting on the task, or by spawning again.

!!! note
    Prefer `device=` over calling [`device!`](@ref KernelAbstractions.device!) inside
    `expr`. A `device!` in the body carries no ordering of its own, so work queued after it
    is ordered neither against the spawning task nor against what the body queued before
    the switch; you would have to bracket it with
    [`record_event`](@ref KernelAbstractions.record_event) and
    [`wait_event`](@ref KernelAbstractions.wait_event) yourself.

Backend authors: see the [notes for backend implementations](@ref implementations_notes)
for the protocol behind these guarantees, and for how to support it without a full
[`synchronize`](@ref).
"""
macro spawn(args...)
    usage = "@spawn expects `@spawn [threadpool] backend [device=id] expr`"
    isempty(args) && throw(ArgumentError(usage))

    # `expr` is always last, so a top-level `=` anywhere before it is our `device=id`
    # argument rather than part of the user's code.
    expr = last(args)
    is_device_arg(x) = Meta.isexpr(x, :(=), 2) && x.args[1] === :device
    is_device_arg(expr) && throw(ArgumentError("$usage; `device=id` must be followed by the expression to run"))

    device = nothing
    positional = Any[]
    for arg in args[1:(end - 1)]
        if is_device_arg(arg)
            device === nothing || throw(ArgumentError("@spawn accepts at most one `device=id` argument"))
            device = arg.args[2]
        else
            push!(positional, arg)
        end
    end

    if length(positional) == 1
        threadpool = nothing
        backend = only(positional)
    elseif length(positional) == 2
        threadpool, backend = positional
    else
        throw(ArgumentError(usage))
    end

    # The whole expansion is escaped so that `Threads.@spawn` is expanded in the caller's
    # scope: that is what lets an enclosing `@sync` see the task, and what makes `$x`
    # interpolation in `expr` work. Our own temporaries are gensyms so they cannot clash
    # with the user's variables.
    b, dev, event, result = gensym(:backend), gensym(:dev), gensym(:event), gensym(:result)
    # `device!` comes first because `wait_event` acts on the queue of the device that is
    # active when it is called: selecting the device afterwards would leave it unordered.
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

    # `device` and the event are both evaluated in the spawning task, so the event captures
    # the work queued on the spawning task's device, not on `dev`.
    return esc(
        quote
            local $b = $backend
            local $dev = $(device === nothing ? :($KI.device($b)) : device)
            local $event = $KI.record_event($b)
            $task
        end
    )
end
