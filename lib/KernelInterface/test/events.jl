# Tests for `record_event`/`wait_event`, the backend side of the protocol behind
# `KernelAbstractions.@spawn`. The protocol is specified in KernelInterface terms, so
# it is replayed here by hand with `Threads.@spawn`: record in the spawning task, then
# `device!`, `wait_event` and `synchronize` in the child task.

# Burns a fixed amount of work per work-item before writing `v`. The accumulator is a
# linear congruential step, which the compiler cannot fold away, and it feeds into the
# store so the loop cannot be dropped.
function slow_fill_kernel(A, v, ::Val{iters}) where {iters}
    i = KI.get_global_id().x
    acc = UInt32(i)
    for k in UInt32(1):UInt32(iters)
        acc = acc * 0x0019660d + k
    end
    if i <= length(A)
        @inbounds A[i] = ifelse(acc == 0x12345678, -v, v)
    end
    return
end

function events_testsuite(backend)
    b = backend()
    dev = KI.device(b)

    # Few work-items and many dependent iterations: tens of milliseconds on a GPU,
    # and still well under a second per launch on a CPU-backed queue.
    N = 64
    iters = Val(2^22)
    slow_fill(A, v) = KI.@kernel b numworkgroups = 1 workgroupsize = N slow_fill_kernel(A, v, iters)

    # Time a launch, after a warm-up that absorbs compilation, and queue enough of
    # them back to back to keep the spawner's queue busy for a couple hundred
    # milliseconds. The minimum of a few runs discards one-off stalls such as a GC
    # pause, which would otherwise inflate the launch count.
    A = KI.zeros(b, Float32, N)
    slow_fill(A, 1.0f0)
    KI.synchronize(b)
    slow_time = minimum(1:3) do _
        @elapsed begin
            slow_fill(A, 1.0f0)
            KI.synchronize(b)
        end
    end
    launches = clamp(ceil(Int, 0.2 / slow_time), 4, 64)

    @testset "ordered across tasks" begin
        # The child queues nothing but the wait, so its `synchronize` can only return
        # once the spawner's queued work has drained. A backend that forgets
        # `wait_event` for its event type fails with a MethodError here, and a
        # `wait_event` that does nothing returns in a few milliseconds. The clock
        # starts before `record_event`, so a backend whose `record_event` is the
        # default full `synchronize` passes just the same. The data check alone would
        # not do: drivers that track hazards between command buffers (Metal, for its
        # default buffers) order the readback after the fills without any wait.
        for v in 1:launches
            slow_fill(A, Float32(v))
        end
        start = time_ns()
        ev = KI.record_event(b)
        task = Threads.@spawn begin
            # `wait_event` acts on the active device's queue, so select it first.
            KI.device!(b, dev)
            KI.wait_event(b, ev)
            KI.synchronize(b)
            elapsed = (time_ns() - start) / 1.0e9
            elapsed, Array(A)
        end
        elapsed, result = fetch(task)
        KI.synchronize(b)
        drained = (time_ns() - start) / 1.0e9
        # Compare against the drain time of this very run rather than the calibration,
        # so a stall during calibration cannot fail a correct backend. Half of it
        # leaves room for a GC pause in the spawner's final `synchronize`.
        @test elapsed >= drained / 2
        @test all(==(Float32(launches)), result)
    end

    if KI.ndevices(b) > 1
        @testset "cross-device" begin
            # `@spawn backend device=id` records on the spawner's device and waits on
            # another one, so a multi-device backend must accept a foreign event. The
            # ordering itself is not observable without peer access; check that the
            # wait is accepted and that work on the other device still runs.
            other = mod1(dev + 1, KI.ndevices(b))
            slow_fill(A, 1.0f0)
            ev = KI.record_event(b)
            task = Threads.@spawn begin
                KI.device!(b, other)
                KI.wait_event(b, ev)
                B = KI.ones(b, Float32, N)
                KI.synchronize(b)
                Array(B)
            end
            @test all(==(1.0f0), fetch(task))
            KI.synchronize(b)
        end
    end
    return nothing
end
