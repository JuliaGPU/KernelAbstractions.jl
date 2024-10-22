using KernelAbstractions
using KernelAbstractions.NDIteration
using InteractiveUtils
using LinearAlgebra
using SparseArrays
using Adapt

identity(x) = x

function unittest_testsuite(Backend, backend_str, backend_mod, BackendArrayT; skip_tests = Set{String}())
    @conditional_testset "partition" skip_tests begin
        backend = Backend()
        let kernel = KernelAbstractions.Kernel{typeof(backend), StaticSize{(64,)}, DynamicSize, typeof(identity)}(backend, identity)
            iterspace, dynamic = KernelAbstractions.partition(kernel, (128,), nothing)
            @test length(blocks(iterspace)) == 2
            @test dynamic isa NoDynamicCheck

            iterspace, dynamic = KernelAbstractions.partition(kernel, (129,), nothing)
            @test length(blocks(iterspace)) == 3
            @test dynamic isa DynamicCheck

            iterspace, dynamic = KernelAbstractions.partition(kernel, (129,), (64,))
            @test length(blocks(iterspace)) == 3
            @test dynamic isa DynamicCheck

            @test_throws ErrorException KernelAbstractions.partition(kernel, (129,), (65,))
            @test KernelAbstractions.backend(kernel) == backend
        end
        let kernel = KernelAbstractions.Kernel{typeof(backend), StaticSize{(64,)}, StaticSize{(128,)}, typeof(identity)}(backend, identity)
            iterspace, dynamic = KernelAbstractions.partition(kernel, (128,), nothing)
            @test length(blocks(iterspace)) == 2
            @test dynamic isa NoDynamicCheck

            iterspace, dynamic = KernelAbstractions.partition(kernel, nothing, nothing)
            @test length(blocks(iterspace)) == 2
            @test dynamic isa NoDynamicCheck

            @test_throws ErrorException KernelAbstractions.partition(kernel, (129,), nothing)
            @test KernelAbstractions.backend(kernel) == backend
        end
    end

    @kernel function index_linear_global(A)
        I = @index(Global, Linear)
        A[I] = I
    end
    @kernel function index_linear_local(A)
        I = @index(Global, Linear)
        i = @index(Local, Linear)
        A[I] = i
    end
    @kernel function index_linear_group(A)
        I = @index(Global, Linear)
        i = @index(Group, Linear)
        A[I] = i
    end
    @kernel function index_cartesian_global(A)
        I = @index(Global, Cartesian)
        A[I] = I
    end
    @kernel function index_cartesian_local(A)
        I = @index(Global, Cartesian)
        i = @index(Local, Cartesian)
        A[I] = i
    end
    @kernel function index_cartesian_group(A)
        I = @index(Global, Cartesian)
        i = @index(Group, Cartesian)
        A[I] = i
    end

    @conditional_testset "get_backend" skip_tests begin
        backend = Backend()
        backendT = typeof(backend).name.wrapper # To look through CUDABackend{true, false}
        @test backend isa backendT

        x = allocate(backend, Float32, 5)
        A = allocate(backend, Float32, 5, 5)
        @test @inferred(KernelAbstractions.get_backend(A)) isa backendT
        @test @inferred(KernelAbstractions.get_backend(view(A, 2:4, 1:3))) isa backendT
        @test @inferred(KernelAbstractions.get_backend(Diagonal(x))) isa backendT
        @test @inferred(KernelAbstractions.get_backend(Tridiagonal(A))) isa backendT
    end

    @conditional_testset "sparse" skip_tests begin
        backend = Backend()
        backendT = typeof(backend).name.wrapper # To look through CUDABackend{true, false}
        @test backend isa backendT

        A = allocate(backend, Float32, 5, 5)
        @test @inferred(KernelAbstractions.get_backend(sparse(A))) isa backendT
    end

    @conditional_testset "adapt" skip_tests begin
        backend = Backend()
        x = allocate(backend, Float32, 5)
        @test adapt(CPU(), x) isa Array
        y = adapt(backend, Array{Float32}(undef, 5))
        @test typeof(y) == typeof(x)
    end

    # TODO: add test for _group and _local_cartesian
    @conditional_testset "indextest" skip_tests begin
        backend = Backend()
        A = allocate(backend, Int, 16, 16)
        index_linear_global(backend, 8)(A, ndrange = length(A))
        synchronize(backend)
        @test all(Array(A) .== LinearIndices(A))

        A = allocate(backend, Int, 8)
        index_linear_local(backend, 8)(A, ndrange = length(A))
        synchronize(backend)
        @test all(Array(A) .== 1:8)

        A = allocate(backend, Int, 16)
        index_linear_local(backend, 8)(A, ndrange = length(A))
        synchronize(backend)
        @test all(Array(A)[1:8] .== 1:8)
        @test all(Array(A)[9:16] .== 1:8)

        A = allocate(backend, Int, 8, 2)
        index_linear_local(backend, 8)(A, ndrange = length(A))
        synchronize(backend)
        @test all(Array(A)[1:8] .== 1:8)
        @test all(Array(A)[9:16] .== 1:8)

        A = allocate(backend, CartesianIndex{2}, 16, 16)
        index_cartesian_global(backend, 8)(A, ndrange = size(A))
        synchronize(backend)
        @test all(Array(A) .== CartesianIndices(A))

        A = allocate(backend, CartesianIndex{1}, 16, 16)
        index_cartesian_global(backend, 8)(A, ndrange = length(A))
        synchronize(backend)
        @test all(Array(A)[:] .== CartesianIndices((length(A),)))

        # Non-multiplies of the workgroupsize
        A = allocate(backend, Int, 7, 7)
        index_linear_global(backend, 8)(A, ndrange = length(A))
        synchronize(backend)
        @test all(Array(A) .== LinearIndices(A))

        A = allocate(backend, Int, 5)
        index_linear_local(backend, 8)(A, ndrange = length(A))
        synchronize(backend)
        @test all(Array(A) .== 1:5)
    end

    @kernel function constarg(A, @Const(B))
        I = @index(Global)
        @inbounds A[I] = B[I]
    end

    @conditional_testset "Const" skip_tests begin
        let kernel = constarg(Backend(), 8, (1024,))
            # this is poking at internals
            iterspace = NDRange{1, StaticSize{(128,)}, StaticSize{(8,)}}()
            ctx = if Backend == CPU
                KernelAbstractions.mkcontext(kernel, 1, nothing, iterspace, Val(NoDynamicCheck()))
            else
                KernelAbstractions.mkcontext(kernel, nothing, iterspace)
            end
            AT = if Backend == CPU
                Array{Float32, 2}
            else
                BackendArrayT{Float32, 2, 1} # AS 1
            end
            IR = sprint() do io
                if backend_str == "CPU"
                    code_llvm(
                        io, kernel.f, (typeof(ctx), AT, AT),
                        optimize = false, raw = true,
                    )
                else
                    backend_mod.code_llvm(
                        io, kernel.f, (typeof(ctx), AT, AT),
                        kernel = true, optimize = true,
                    )
                end
            end
            if backend_str == "CPU"
                @test occursin("!alias.scope", IR)
                @test occursin("!noalias", IR)
            elseif backend_str == "CUDA"
                @test occursin("@llvm.nvvm.ldg", IR)
            elseif backend_str == "ROCM"
                @test occursin("addrspace(4)", IR)
            else
                @test_skip false
            end
        end
    end

    @kernel function kernel_val!(a, ::Val{m}) where {m}
        I = @index(Global)
        @inbounds a[I] = m
    end

    A = KernelAbstractions.zeros(Backend(), Int64, 1024)
    kernel_val!(Backend())(A, Val(3), ndrange = size(A))
    synchronize(Backend())
    @test all((a) -> a == 3, Array(A))

    @kernel function kernel_empty()
        nothing
    end

    @conditional_testset "CPU synchronization" skip_tests begin
        kernel_empty(CPU(), 1)(ndrange = 1)
        synchronize(CPU())
    end

    @conditional_testset "Zero iteration space $Backend" skip_tests begin
        kernel_empty(Backend(), 1)(ndrange = 1)
        kernel_empty(Backend(), 1)(ndrange = 0)
        synchronize(Backend())

        kernel_empty(Backend(), 1)(ndrange = 0)
        synchronize(Backend())
    end


    @conditional_testset "return statement" skip_tests begin
        try
            @eval @kernel function kernel_return()
                return
            end
            @test false
        catch e
            @test e.error ==
                ErrorException("Return statement not permitted in a kernel function kernel_return")
        end
    end

    @conditional_testset "fallback test: callable types" skip_tests begin
        @eval begin
            function f end
            @kernel function (a::typeof(f))(x, ::Val{m}) where {m}
                I = @index(Global)
                @inbounds x[I] = m
            end
            @kernel function (a::typeof(f))(x, ::Val{1})
                I = @index(Global)
                @inbounds x[I] = 1
            end
            x = [1, 2, 3]
            env = f(CPU())(x, Val(4); ndrange = length(x))
            synchronize(CPU())
            @test x == [4, 4, 4]

            x = [1, 2, 3]
            env = f(CPU())(x, Val(1); ndrange = length(x))
            synchronize(CPU())
            @test x == [1, 1, 1]
        end
    end

    @conditional_testset "priority" skip_tests begin
        KernelAbstractions.priority!(Backend(), :normal)
        KernelAbstractions.priority!(Backend(), :high)
        KernelAbstractions.priority!(Backend(), :low)

        @test_throws ErrorException KernelAbstractions.priority!(Backend(), :default)
    end

    function f(KernelAbstractions.@context, a)
        I = @index(Global, Linear)
        a[I] = 1
        return
    end
    @kernel cpu = false function context_kernel(a)
        f(KernelAbstractions.@context, a)
    end

    @testset "No CPU kernel" begin
        if !(Backend() isa CPU)
            A = KernelAbstractions.zeros(Backend(), Int64, 1024)
            context_kernel(Backend())(A, ndrange = size(A))
            synchronize(Backend())
            @test all((a) -> a == 1, Array(A))
        else
            @test_throws ErrorException("This kernel is unavailable for backend CPU") context_kernel(Backend())
        end
    end

    @testset "functional" begin
        @test KernelAbstractions.functional(Backend()) isa Union{Missing, Bool}
    end

    @testset "CPU default workgroupsize" begin
        @test KernelAbstractions.default_cpu_workgroupsize((64,)) == (64,)
        @test KernelAbstractions.default_cpu_workgroupsize((1024,)) == (1024,)
        @test KernelAbstractions.default_cpu_workgroupsize((2056,)) == (1024,)
        @test KernelAbstractions.default_cpu_workgroupsize((64, 64)) == (64, 16)
        @test KernelAbstractions.default_cpu_workgroupsize((64, 64, 64, 4)) == (64, 16, 1, 1)
        @test KernelAbstractions.default_cpu_workgroupsize((64, 15)) == (64, 15)
        @test KernelAbstractions.default_cpu_workgroupsize((5, 7, 13, 17)) == (5, 7, 13, 2)
    end

    @testset "empty arrays" begin
        backend = Backend()
        @test size(allocate(backend, Float32, 0)) == (0,)
        @test size(allocate(backend, Float32, 3, 0)) == (3, 0)
        @test size(allocate(backend, Float32, 0, 9)) == (0, 9)
        @test size(KernelAbstractions.zeros(backend, Float32, 0)) == (0,)
        @test size(KernelAbstractions.zeros(backend, Float32, 3, 0)) == (3, 0)
        @test size(KernelAbstractions.zeros(backend, Float32, 0, 9)) == (0, 9)
    end

    return
end
