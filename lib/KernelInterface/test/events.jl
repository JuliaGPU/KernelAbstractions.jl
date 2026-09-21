# Tests for `record_event`/`wait_event`, the backend side of the protocol behind
# `KernelAbstractions.@spawn`. The protocol is specified in KernelInterface terms, so
# it is replayed here by hand with `Threads.@spawn`: record in the spawning task, then
# `device!`, `wait_event` and `synchronize` in the child task.

# Burns `iters` dependent steps per work-item before writing `v`. The accumulator is
# a linear congruential step, which the compiler cannot fold away, and it feeds into
# the store so the loop cannot be dropped. `iters` is a run-time argument so the
# duration can be tuned below without recompiling.
function slow_fill_kernel(A, v, iters::UInt32)
    i = KI.get_global_id().x
    acc = UInt32(i)
    for k in UInt32(1):iters
        acc = acc * 0x19660d + k
    end
    if i <= length(A)
        @inbounds A[i] = ifelse(acc == 0x12345678, -v, v)
    end
    return
end

function events_testsuite(backend)
    b = backend()
    dev = KI.device(b)

    N = 64
    A = KI.zeros(b, Float32, N)
    slow_fill(v, iters) = KI.@kernel b numworkgroups = 1 workgroupsize = N slow_fill_kernel(A, v, UInt32(iters))

    # Time a launch as the minimum of a few runs: a backend's `synchronize` may run a
    # GC or otherwise stall once in a while, and the minimum discards that.
    function time_launch(iters)
        return minimum(1:3) do _
            @elapsed begin
                slow_fill(1.0f0, iters)
                KI.synchronize(b)
            end
        end
    end

    # Tune the kernel to about 10ms per launch, after a warm-up that absorbs
    # compilation, and queue enough launches for a couple hundred milliseconds.
    base = 2^20
    time_launch(base)
    iters = clamp(round(Int, base * 0.01 / time_launch(base)), base, 2^30)
    launches = 20
    expected = launches * time_launch(iters)

    @testset "ordered across tasks" begin
        # The child queues nothing but the wait, so its `synchronize` can only return
        # once the spawner's queued work has drained. A backend that forgets
        # `wait_event` for its event type fails with a MethodError here, and a
        # `wait_event` that does nothing returns in a few milliseconds. The clock
        # starts before `record_event`, so a backend whose `record_event` is the
        # default full `synchronize` passes just the same. The data check alone would
        # not do: drivers that track hazards between command buffers (Metal, for its
        # default buffers) order the readback after the fills without any wait.
        #
        # Collect beforehand so that a GC pause is unlikely to land inside the
        # measurement and mask a missing wait.
        GC.gc()
        for v in 1:launches
            slow_fill(Float32(v), iters)
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
        # A third of the calibrated drain time leaves room for the device clocking up
        # between calibration and this run; a missing wait is far below that.
        @test elapsed >= expected / 3
        @test all(==(Float32(launches)), result)
    end

    if KI.ndevices(b) > 1
        @testset "cross-device" begin
            # `@spawn backend device=id` records on the spawner's device and waits on
            # another one, so a multi-device backend must accept a foreign event. The
            # ordering itself is not observable without peer access; check that the
            # wait is accepted and that work on the other device still runs.
            other = mod1(dev + 1, KI.ndevices(b))
            slow_fill(1.0f0, iters)
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
