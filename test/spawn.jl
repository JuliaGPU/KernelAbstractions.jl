@kernel function spawn_fill_kernel(A, v)
    I = @index(Global)
    @inbounds A[I] = v
end

# A single work item runs `n` dependent multiply-adds before writing, so one launch
# keeps the queue busy for tens of milliseconds. The chain converges to exactly `2v`.
@kernel function spawn_slow_fill_kernel(A, v, n)
    I = @index(Global)
    x = zero(v)
    for _ in 1:n
        x = muladd(x, 0.5f0, v)
    end
    @inbounds A[I] = x
end

# The spawned task queues nothing itself, so `wait(task)` can only take as long as the
# parent's queued work if `wait_event` ordered the new queue after it.
function spawn_ordered_elapsed(backend, A, slow, n)
    t = @elapsed begin
        slow(A, 1.5f0, n, ndrange = 1)   # queued, not synchronized
        wait(KernelAbstractions.@spawn backend nothing)
    end
    KernelAbstractions.synchronize(backend)   # do not let a still-running kernel skew the next run
    return t
end

function spawn_testsuite(Backend, AT)
    backend = Backend()

    @testset "ordered after the spawning task" begin
        # Sharing data between the tasks would not do: backends that track which queue
        # owns an array (CUDA) or which command buffers touch a buffer (Metal) order the
        # accesses themselves, with or without `wait_event`. Observe the ordering through
        # time instead.
        A = KernelAbstractions.zeros(backend, Float32, 1)
        slow = spawn_slow_fill_kernel(backend, 1)
        # Reference: the minimum of a few launches, so a stall or a device still clocking
        # up does not inflate it. The chain grows until the kernel takes a few
        # milliseconds, so that it stands well clear of the spawn overhead on any device.
        function time_slow(n)
            slow(A, 1.5f0, n, ndrange = 1)   # compile up front
            KernelAbstractions.synchronize(backend)
            return minimum(1:3) do _
                @elapsed begin
                    slow(A, 1.5f0, n, ndrange = 1)
                    KernelAbstractions.synchronize(backend)
                end
            end
        end
        n = 2^24
        t_slow = time_slow(n)
        while t_slow < 0.005 && n < 2^32
            n *= 4
            t_slow = time_slow(n)
        end
        @test t_slow > 0.005
        @test all(==(3.0f0), Array(A))

        # The first round-trip also pays for compiling the task body, so it is not
        # telling: warm up once, then measure. Without the wait the warmed round-trip is
        # well under a millisecond.
        spawn_ordered_elapsed(backend, A, slow, n)
        @test spawn_ordered_elapsed(backend, A, slow, n) >= t_slow / 2
    end

    @testset "results visible after wait" begin
        A = KernelAbstractions.zeros(backend, Float32, 256)
        fill = spawn_fill_kernel(backend, 32)
        task = KernelAbstractions.@spawn backend fill(A, 3.0f0, ndrange = length(A))
        wait(task)
        @test all(==(3.0f0), Array(A))
    end

    @testset "task properties" begin
        dev = KernelAbstractions.device(backend)
        task = KernelAbstractions.@spawn backend KernelAbstractions.device(backend)
        @test fetch(task) == dev

        # `Threads.@spawn` only honors a pool that has threads, and falls back to
        # `:default` otherwise, which is the case unless Julia was started with
        # interactive threads (the default from 1.12 on).
        interactive = Threads.nthreads(:interactive) > 0 ? :interactive : :default
        task = KernelAbstractions.@spawn :interactive backend Threads.threadpool()
        @test fetch(task) === interactive

        pool = :default
        task = KernelAbstractions.@spawn pool backend Threads.threadpool()
        @test fetch(task) === :default

        # The backend expression is evaluated once, in the spawning task.
        counter = Ref(0)
        get_backend_once() = (counter[] += 1; backend)
        wait(KernelAbstractions.@spawn get_backend_once() nothing)
        @test counter[] == 1

        # Errors propagate through `wait`/`fetch` like they do for `Threads.@spawn`.
        task = KernelAbstractions.@spawn backend error("boom")
        @test_throws TaskFailedException wait(task)

        # `$x` captures the value at spawn time, like it does for `Threads.@spawn`.
        x = Ref(1)
        task = KernelAbstractions.@spawn backend $(x[]) + 1
        x[] = 10
        @test fetch(task) == 2
    end

    @testset "device" begin
        # Without `device=`, the task inherits the spawning task's device.
        dev = KernelAbstractions.device(backend)
        @test fetch(KernelAbstractions.@spawn backend KernelAbstractions.device(backend)) == dev

        # `device=` selects the device explicitly, as a literal or an expression.
        task = KernelAbstractions.@spawn backend device = dev KernelAbstractions.device(backend)
        @test fetch(task) == dev
        task = KernelAbstractions.@spawn backend device = 1 KernelAbstractions.device(backend)
        @test fetch(task) == 1

        # `device=` combines with a threadpool, and the kernels still run.
        A = KernelAbstractions.zeros(backend, Float32, 128)
        fill = spawn_fill_kernel(backend, 32)
        wait(KernelAbstractions.@spawn :default backend device = dev fill(A, 7.0f0, ndrange = length(A)))
        @test all(==(7.0f0), Array(A))

        # An out-of-range device fails inside the task, as `device!` would.
        nd = KernelAbstractions.ndevices(backend)
        @test_throws TaskFailedException wait(KernelAbstractions.@spawn backend device = nd + 1 nothing)

        # A top-level assignment in the body is a body, not a `device=` argument.
        @test fetch(KernelAbstractions.@spawn backend y = 41 + 1) == 42
    end

    @testset "@sync" begin
        # An enclosing `@sync` waits for the task, and sees its errors.
        A = KernelAbstractions.zeros(backend, Float32, 256)
        fill = spawn_fill_kernel(backend, 32)
        done = Ref(false)
        @sync begin
            KernelAbstractions.@spawn backend begin
                fill(A, 5.0f0, ndrange = length(A))
                done[] = true
            end
        end
        @test done[]
        @test all(==(5.0f0), Array(A))
        @test_throws CompositeException @sync begin
            KernelAbstractions.@spawn backend error("boom")
        end
    end

    @testset "many tasks" begin
        n = 8
        arrays = [KernelAbstractions.zeros(backend, Float32, 128) for _ in 1:n]
        fill = spawn_fill_kernel(backend, 32)
        tasks = map(1:n) do i
            KernelAbstractions.@spawn backend fill(arrays[i], Float32(i), ndrange = 128)
        end
        foreach(wait, tasks)
        for i in 1:n
            @test all(==(Float32(i)), Array(arrays[i]))
        end
    end
    return
end
