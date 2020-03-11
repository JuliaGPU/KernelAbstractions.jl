using KernelAbstractions
using KernelAbstractions.NDIteration
using CUDAapi
using InteractiveUtils
if has_cuda_gpu()
    using CuArrays
    using CUDAnative
    CuArrays.allowscalar(false)
end

identity(x)=x
@testset "partition" begin
    let kernel = KernelAbstractions.Kernel{CPU, StaticSize{(64,)}, DynamicSize, typeof(identity)}(identity)
        iterspace, dynamic = KernelAbstractions.partition(kernel, (128,), nothing)
        @test length(blocks(iterspace)) == 2
        @test !dynamic

        iterspace, dynamic = KernelAbstractions.partition(kernel, (129,), nothing)
        @test length(blocks(iterspace)) == 3
        @test dynamic

        iterspace, dynamic = KernelAbstractions.partition(kernel, (129,), (64,))
        @test length(blocks(iterspace)) == 3
        @test dynamic

        @test_throws ErrorException KernelAbstractions.partition(kernel, (129,), (65,))
    end
    let kernel = KernelAbstractions.Kernel{CPU, StaticSize{(64,)}, StaticSize{(128,)}, typeof(identity)}(identity)
        iterspace, dynamic = KernelAbstractions.partition(kernel, (128,), nothing)
        @test length(blocks(iterspace)) == 2
        @test !dynamic

        iterspace, dynamic = KernelAbstractions.partition(kernel, nothing, nothing)
        @test length(blocks(iterspace)) == 2
        @test !dynamic

        @test_throws ErrorException KernelAbstractions.partition(kernel, (129,), nothing)
    end
end

@kernel function index_linear_global(A)
       I = @index(Global, Linear)
       A[I] = I
end
@kernel function index_linear_local(A)
       I  = @index(Global, Linear)
       i = @index(Local, Linear)
       A[I] = i
end
@kernel function index_linear_group(A)
       I  = @index(Global, Linear)
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

function indextest(backend, ArrayT)
    # TODO: add test for _group and _local_cartesian
    A = ArrayT{Int}(undef, 16, 16)
    wait(index_linear_global(backend, 8)(A, ndrange=length(A)))
    @test all(A .== LinearIndices(A))

    A = ArrayT{Int}(undef, 8)
    wait(index_linear_local(backend, 8)(A, ndrange=length(A)))
    @test all(A .== 1:8)

    A = ArrayT{Int}(undef, 16)
    wait(index_linear_local(backend, 8)(A, ndrange=length(A)))
    @test all(A[1:8] .== 1:8)
    @test all(A[9:16] .== 1:8)

    A = ArrayT{Int}(undef, 8, 2)
    wait(index_linear_local(backend, 8)(A, ndrange=length(A)))
    @test all(A[1:8] .== 1:8)
    @test all(A[9:16] .== 1:8)

    A = ArrayT{CartesianIndex{2}}(undef, 16, 16)
    wait(index_cartesian_global(backend, 8)(A, ndrange=size(A)))
    @test all(A .== CartesianIndices(A))

    A = ArrayT{CartesianIndex{1}}(undef, 16, 16)
    wait(index_cartesian_global(backend, 8)(A, ndrange=length(A)))
    @test all(A[:] .== CartesianIndices((length(A),)))

    # Non-multiplies of the workgroupsize
    A = ArrayT{Int}(undef, 7, 7)
    wait(index_linear_global(backend, 8)(A, ndrange=length(A)))
    @test all(A .== LinearIndices(A))

    A = ArrayT{Int}(undef, 5)
    wait(index_linear_local(backend, 8)(A, ndrange=length(A)))
    @test all(A .== 1:5)
end

@testset "indextest" begin
    indextest(CPU(), Array)
    if has_cuda_gpu()
        indextest(CUDA(), CuArray)
    end
end

@kernel function constarg(A, @Const(B))
    I = @index(Global)
    @inbounds A[I] = B[I]
end

@testset "Const" begin
    let kernel = constarg(CPU(), 8, (1024,))
        # this is poking at internals
        iterspace = NDRange{1, StaticSize{(128,)}, StaticSize{(8,)}}();
        ctx = KernelAbstractions.mkcontext(kernel, 1, nothing, iterspace, Val(false))
        AT = Array{Float32, 2}
        IR = sprint() do io
            code_llvm(io, KernelAbstractions.Cassette.overdub, 
                     (typeof(ctx), typeof(kernel.f), AT, AT),
                     optimize=false, raw=true)
        end
        @test occursin("!alias.scope", IR)
        @test occursin("!noalias", IR)
    end

    if has_cuda_gpu()
        let kernel = constarg(CUDA(), 8, (1024,))
            # this is poking at internals
            iterspace = NDRange{1, StaticSize{(128,)}, StaticSize{(8,)}}();
            ctx = KernelAbstractions.mkcontext(kernel, nothing, iterspace)
            AT = CUDAnative.CuDeviceArray{Float32, 2, CUDAnative.AS.Global}
            IR = sprint() do io
                CUDAnative.code_llvm(io, KernelAbstractions.Cassette.overdub, 
                        (typeof(ctx), typeof(kernel.f), AT, AT),
                        kernel=true, optimize=false)
            end
            @test occursin("@llvm.nvvm.ldg", IR)
        end
    end
end

@kernel function kernel_val!(a, ::Val{m}) where {m}
    I = @index(Global)
    @inbounds a[I] = m
end

A = zeros(Int64, 1024)
wait(kernel_val!(CPU())(A,Val(3), ndrange=size(A)))
@test all((a)->a==3, A)
if has_cuda_gpu()
    A = CuArrays.zeros(Int64, 1024)
    wait(kernel_val!(CUDA())(A,Val(3), ndrange=size(A)))
    @test all((a)->a==3, A)
end

@kernel function kernel_empty()
    return
end
if has_cuda_gpu()
    @testset "CPU--CUDA dependencies" begin
        event1 = kernel_empty(CPU(), 1)(ndrange=1)
        event2 = kernel_empty(CUDA(), 1)(ndrange=1)
        event3 = kernel_empty(CPU(), 1)(ndrange=1)
        event4 = kernel_empty(CUDA(), 1)(ndrange=1)
        event5 = kernel_empty(CUDA(), 1)(ndrange=1, dependencies=(event1, event2, event3, event4))
        wait(event5)
        @test event5 isa KernelAbstractions.Event

        event1 = kernel_empty(CPU(), 1)(ndrange=1)
        event2 = kernel_empty(CUDA(), 1)(ndrange=1)
        event3 = kernel_empty(CPU(), 1)(ndrange=1)
        event4 = kernel_empty(CUDA(), 1)(ndrange=1)
        event5 = kernel_empty(CPU(), 1)(ndrange=1, dependencies=(event1, event2, event3, event4))
        wait(event5)
        @test event5 isa KernelAbstractions.Event
    end
    @testset "CUDA wait" begin
        event = kernel_empty(CUDA(), 1)(ndrange=1)
        wait(CUDA(), event)
        @test event isa KernelAbstractions.Event
    end
end

@testset "CPU dependencies" begin
    event = Event(CPU())
    event = kernel_empty(CPU(), 1)(ndrange=1, dependencies=(event))
    wait(event)
end

@testset "fallback test: callable types" begin
    struct A end
    @kernel function (a::A)(x, ::Val{m}) where m
        I = @index(Global)
        @inbounds x[I] = m
    end
    x = [1,2,3]
    env = A()(CPU())(x, Val(4); ndrange=length(x))
    wait(env)
    @test x == [4,4,4]
end
