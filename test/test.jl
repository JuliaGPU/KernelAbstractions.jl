using KernelAbstractions
using KernelAbstractions.NDIteration
using CUDA
using InteractiveUtils


if has_cuda_gpu()
    CUDA.allowscalar(false)
end

identity(x) = x

@testset "partition" begin
    let kernel = KernelAbstractions.Kernel{CPU, StaticSize{(64,)}, DynamicSize, typeof(identity)}(identity)
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
    end
    let kernel = KernelAbstractions.Kernel{CPU, StaticSize{(64,)}, StaticSize{(128,)}, typeof(identity)}(identity)
        iterspace, dynamic = KernelAbstractions.partition(kernel, (128,), nothing)
        @test length(blocks(iterspace)) == 2
        @test dynamic isa NoDynamicCheck

        iterspace, dynamic = KernelAbstractions.partition(kernel, nothing, nothing)
        @test length(blocks(iterspace)) == 2
        @test dynamic isa NoDynamicCheck

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
        indextest(CUDADevice(), CuArray)
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
        ctx = KernelAbstractions.mkcontext(kernel, 1, nothing, iterspace, Val(NoDynamicCheck()))
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
        let kernel = constarg(CUDADevice(), 8, (1024,))
            # this is poking at internals
            iterspace = NDRange{1, StaticSize{(128,)}, StaticSize{(8,)}}();
            ctx = KernelAbstractions.mkcontext(kernel, nothing, iterspace)
            AT = CUDA.CuDeviceArray{Float32, 2, CUDA.AS.Global}
            IR = sprint() do io
                CUDA.code_llvm(io, KernelAbstractions.Cassette.overdub,
                               (typeof(ctx), typeof(kernel.f), AT, AT),
                               kernel=true, optimize=true)
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
    A = CUDA.zeros(Int64, 1024)
    wait(kernel_val!(CUDADevice())(A,Val(3), ndrange=size(A)))
    @test all((a)->a==3, A)
end

@kernel function kernel_empty()
    nothing
end
if has_cuda_gpu()
    @testset "CPU--CUDA dependencies" begin
        event1 = kernel_empty(CPU(), 1)(ndrange=1)
        event2 = kernel_empty(CUDADevice(), 1)(ndrange=1)
        event3 = kernel_empty(CPU(), 1)(ndrange=1)
        event4 = kernel_empty(CUDADevice(), 1)(ndrange=1)
        @test_throws ErrorException event5 = kernel_empty(CUDADevice(), 1)(ndrange=1, dependencies=(event1, event2, event3, event4))
        # wait(event5)
        # @test event5 isa KernelAbstractions.Event

        event1 = kernel_empty(CPU(), 1)(ndrange=1)
        event2 = kernel_empty(CUDADevice(), 1)(ndrange=1)
        event3 = kernel_empty(CPU(), 1)(ndrange=1)
        event4 = kernel_empty(CUDADevice(), 1)(ndrange=1)
        event5 = kernel_empty(CPU(), 1)(ndrange=1, dependencies=(event1, event2, event3, event4))
        wait(event5)
        @test event5 isa KernelAbstractions.Event
    end
    @testset "CUDA wait" begin
        event = kernel_empty(CUDADevice(), 1)(ndrange=1)
        wait(CUDADevice(), event)
        @test event isa KernelAbstractions.Event
    end
end

@testset "CPU dependencies" begin
    event = Event(CPU())
    event = kernel_empty(CPU(), 1)(ndrange=1, dependencies=(event))
    wait(event)
end

@testset "MultiEvent" begin
  event1 = kernel_empty(CPU(), 1)(ndrange=1)
  event2 = kernel_empty(CPU(), 1)(ndrange=1)
  event3 = kernel_empty(CPU(), 1)(ndrange=1)

  @test MultiEvent(nothing) isa Event
  @test MultiEvent((MultiEvent(nothing),)) isa Event
  @test MultiEvent(event1) isa Event
  @test MultiEvent((event1, event2, event3)) isa Event
end

if has_cuda_gpu()
  @testset "MultiEvent CUDA" begin
    event1 = kernel_empty(CUDADevice(), 1)(ndrange=1)
    event2 = kernel_empty(CPU(), 1)(ndrange=1)
    event3 = kernel_empty(CUDADevice(), 1)(ndrange=1)

    @test MultiEvent(event1) isa Event
    @test MultiEvent((event1, event2, event3)) isa Event
  end
end

@testset "Zero iteration space" begin
    event1 = kernel_empty(CPU(), 1)(ndrange=1)
    event2 = kernel_empty(CPU(), 1)(ndrange=0; dependencies=event1)
    @test event2 == MultiEvent(event1)
    event = kernel_empty(CPU(), 1)(ndrange=0)
    @test event == MultiEvent(nothing)
end


if has_cuda_gpu()
    @testset "Zero iteration space CUDA" begin
        event1 = kernel_empty(CUDADevice(), 1)(ndrange=1)
        event2 = kernel_empty(CUDADevice(), 1)(ndrange=0; dependencies=event1)
        @test event2 == MultiEvent(event1)
        event = kernel_empty(CUDADevice(), 1)(ndrange=0)
        @test event == MultiEvent(nothing)
    end
end

@testset "return statement" begin
    try
        @eval @kernel function kernel_return()
            return
        end
    catch e
        @test e.error ==
            ErrorException("Return statement not permitted in a kernel function kernel_return")
    end
end

@testset "fallback test: callable types" begin
    function f end
    @kernel function (a::typeof(f))(x, ::Val{m}) where m
        I = @index(Global)
        @inbounds x[I] = m
    end
    @kernel function (a::typeof(f))(x, ::Val{1})
        I = @index(Global)
        @inbounds x[I] = 1
    end
    x = [1,2,3]
    env = f(CPU())(x, Val(4); ndrange=length(x))
    wait(env)
    @test x == [4,4,4]

    x = [1,2,3]
    env = f(CPU())(x, Val(1); ndrange=length(x))
    wait(env)
    @test x == [1,1,1]
end

@testset "special functions: gamma" begin
    import SpecialFunctions

    @kernel function gamma_knl(A, @Const(B))
        I = @index(Global)
        @inbounds A[I] = SpecialFunctions.gamma(B[I])
    end

    x = [1.0,2.0,3.0,5.5]
    y = similar(x)
    event = gamma_knl(CPU())(y, x; ndrange=length(x))
    wait(event)
    @test y == SpecialFunctions.gamma.(x)

    if has_cuda_gpu()
        cx = CuArray(x)
        cy = similar(cx)
        event = gamma_knl(CUDADevice())(cy, cx; ndrange=length(x))
        wait(event)

        cy = Array(cy)
        @test cy[1:3] == SpecialFunctions.gamma.(x[1:3])
        @test cy[4] ≈ SpecialFunctions.gamma.(x[4])
    end
end

@testset "special functions: erf" begin
    import SpecialFunctions

    @kernel function erf_knl(A, @Const(B))
        I = @index(Global)
        @inbounds A[I] = SpecialFunctions.erf(B[I])
    end

    x = [-1.0,-0.5,0.0,1e-3,1.0,2.0,5.5]
    y = similar(x)
    event = erf_knl(CPU())(y, x; ndrange=length(x))
    wait(event)
    @test y == SpecialFunctions.erf.(x)

    if has_cuda_gpu()
        cx = CuArray(x)
        cy = similar(cx)
        event = erf_knl(CUDADevice())(cy, cx; ndrange=length(x))
        wait(event)

        cy = Array(cy)
        @test cy[1:3] == SpecialFunctions.erf.(x[1:3])
        @test cy[4] ≈ SpecialFunctions.erf.(x[4])
    end
end

@testset "special functions: erfc" begin
    import SpecialFunctions

    @kernel function erfc_knl(A, @Const(B))
        I = @index(Global)
        @inbounds A[I] = SpecialFunctions.erfc(B[I])
    end

    x = [-1.0,-0.5,0.0,1e-3,1.0,2.0,5.5]
    y = similar(x)
    event = erfc_knl(CPU())(y, x; ndrange=length(x))
    wait(event)
    @test y == SpecialFunctions.erfc.(x)

    if has_cuda_gpu()
        cx = CuArray(x)
        cy = similar(cx)
        event = erfc_knl(CUDADevice())(cy, cx; ndrange=length(x))
        wait(event)

        cy = Array(cy)
        @test cy[1:3] == SpecialFunctions.erfc.(x[1:3])
        @test cy[4] ≈ SpecialFunctions.erfc.(x[4])
    end
end
