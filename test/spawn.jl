@kernel function spawn_mul2_kernel(A)
    I = @index(Global)
    @inbounds A[I] = 2 * A[I]
end

@kernel function spawn_fill_kernel(A, v)
    I = @index(Global)
    @inbounds A[I] = v
end

function spawn_testsuite(Backend, AT)
    backend = Backend()

    @testset "ordered after the spawning task" begin
        A = KernelAbstractions.ones(backend, Float32, 1024)
        mul2 = spawn_mul2_kernel(backend, 64)
        mul2(A, ndrange = length(A))   # queued by this task, not synchronized
        task = KernelAbstractions.@spawn backend begin
            mul2(A, ndrange = length(A))
            mul2(A, ndrange = length(A))
            :done
        end
        @test task isa Task
        @test fetch(task) === :done
        @test all(==(8.0f0), Array(A))
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
