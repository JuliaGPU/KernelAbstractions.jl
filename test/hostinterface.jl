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
        @test KI.sub_group_size(backend) isa Int
        @test KI.shfl_down_types(backend) isa Vector{DataType}

        function ki_hostinterface_kernel(x)
            i = KI.get_global_id().x
            if i <= length(x)
                @inbounds x[i] = 1
            end
            return
        end

        x = AT(zeros(Float32, 4))
        kernel = KI.@kernel _backend() launch = false ki_hostinterface_kernel(x)
        @test kernel isa KI.Kernel
        @test KI.kernel_max_work_group_size(kernel) isa Int
        @test KI.kernel_max_work_group_size(kernel; max_work_items = 1) == 1
    end

    return nothing
end
