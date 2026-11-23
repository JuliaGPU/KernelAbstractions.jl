@kernel function graph_axpy_kernel!(y, @Const(x), a)
    i = @index(Global)
    @inbounds y[i] += a * x[i]
end

function hostinterface_testsuite(_backend, AT)
    backend = _backend()

    @testset "capability queries" begin
        @test KernelAbstractions.supports_unified(backend) isa Bool
        @test KernelAbstractions.supports_atomics(backend) isa Bool
        @test KernelAbstractions.supports_float64(backend) isa Bool
        @test KernelAbstractions.functional(backend) isa Union{Missing, Bool}
    end

    @testset "device management" begin
        @test KernelAbstractions.device(backend) isa Int
        @test KernelAbstractions.ndevices(backend) isa Int
        KernelAbstractions.device!(backend, KernelAbstractions.device(backend))
        @test_throws ArgumentError KernelAbstractions.device!(backend, 0)
        @test_throws ArgumentError KernelAbstractions.device!(backend, KernelAbstractions.ndevices(backend) + 1)
    end

    @testset "priority!" begin
        KernelAbstractions.priority!(backend, :normal)
        KernelAbstractions.priority!(backend, :high)
        KernelAbstractions.priority!(backend, :low)
        @test_throws ErrorException KernelAbstractions.priority!(backend, :bogus)
    end

    @testset "allocation" begin
        A = KernelAbstractions.allocate(backend, Float32, 8)
        @test A isa AT{Float32, 1}
        @test size(A) == (8,)

        Z = KernelAbstractions.zeros(backend, Float32, 4, 4)
        @test all(iszero, Array(Z))

        O = KernelAbstractions.ones(backend, Float32, 4, 4)
        @test all(isone, Array(O))

        if KernelAbstractions.supports_unified(backend)
            U = KernelAbstractions.allocate(backend, Float32, 4; unified = true)
            @test U isa AbstractArray{Float32}
        else
            @test_throws Exception KernelAbstractions.allocate(backend, Float32, 4; unified = true)
        end
    end

    @testset "get_backend" begin
        A = KernelAbstractions.allocate(backend, Float32, 4)
        @test KernelAbstractions.get_backend(A) isa Backend
    end

    @testset "pagelock!" begin
        A = Vector{Float32}(undef, 4)
        @test KernelAbstractions.pagelock!(backend, A) isa Union{Missing, Nothing}
    end


    @testset "KernelInterface host functions" begin
        @test KI.max_work_group_size(backend) isa Int
        @test KI.multiprocessor_count(backend) isa Int
        @test KI.supports_subgroups(backend) isa Bool
        if KI.supports_subgroups(backend)
            @test KI.sub_group_size(backend) isa Int
        end
        @test KI.supports_shuffle(backend, Float32) isa Bool

        function ki_hostinterface_kernel(x)
            i = KI.get_global_id().x
            if i <= length(x)
                @inbounds x[i] = 1
            end
            return
        end

        x = AT(zeros(Float32, 4))
        kernel = KI.@launch _backend() launch = false ki_hostinterface_kernel(x)
        @test kernel isa KI.Kernel
        @test KI.max_work_group_size(kernel) isa Int
        @test KI.launch_configuration(kernel) isa @NamedTuple{workgroupsize::Int}
        @test KI.launch_configuration(kernel; max_work_group_size = 1).workgroupsize == 1
    end

    # Graphs are optional, so these results have to hold whether or not the backend
    # records the work: the function follows the contract of `capture`.
    @testset "graphs" begin
        @test KernelAbstractions.supports_graphs(backend) isa Bool
        strict = KernelAbstractions.supports_graphs(backend)
        N = 64
        x = AT(ones(Float32, N))
        y = KernelAbstractions.zeros(backend, Float32, N)
        z = KernelAbstractions.zeros(backend, Float32, N)
        axpy! = graph_axpy_kernel!(backend)
        work = () -> (axpy!(y, x, 1.0f0; ndrange = N); axpy!(y, x, 2.0f0; ndrange = N))

        graph = KernelAbstractions.capture(work, backend; strict)
        @test graph isa KernelAbstractions.Graph
        @test KernelAbstractions.is_captured(graph) == strict
        synchronize(backend)
        @test all(==(3), Array(y))  # the warm-up executed the work once

        for _ in 1:3
            graph()
        end
        synchronize(backend)
        @test all(==(12), Array(y))

        # the contents of the arrays may change between calls
        twos = fill(2.0f0, N)
        KernelAbstractions.copyto!(backend, x, twos)
        synchronize(backend)
        graph()
        synchronize(backend)
        @test all(==(18), Array(y))

        # other arrays and scalars need an update, which doesn't execute anything
        KernelAbstractions.update!(graph, () -> axpy!(z, x, 0.5f0; ndrange = N); strict)
        synchronize(backend)
        @test all(iszero, Array(z))
        graph()
        synchronize(backend)
        @test all(==(1), Array(z))
        @test all(==(18), Array(y))

        # `@captured` executes the work once per call, whatever is recorded
        fill!(z, 0)
        for a in 1.0f0:3.0f0
            KernelAbstractions.@captured backend axpy!(z, x, a; ndrange = N)
        end
        synchronize(backend)
        @test all(==(12), Array(z))
    end

    return nothing
end
