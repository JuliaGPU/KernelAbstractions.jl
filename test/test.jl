using KernelAbstractions
using CUDAapi
using InteractiveUtils
if has_cuda_gpu()
    using CuArrays
    using CUDAnative
    CuArrays.allowscalar(false)
end

import KernelAbstractions: StaticSize, DynamicSize

@testset "CompilerMetadata constructors" begin
    # cpu static
    let cm =  KernelAbstractions.CompilerMetadata{StaticSize{(64,)}, StaticSize{(128,)}, false}(1, nothing, nothing)
        @test KernelAbstractions.__ndrange(cm) isa CartesianIndices
        @test KernelAbstractions.__ndrange(cm) == CartesianIndices((128,))
        @test KernelAbstractions.__groupsize(cm) == 64
        @test KernelAbstractions.__groupindex(cm) == 1
        @test KernelAbstractions.__dynamic_checkbounds(cm) == false
    end

    # gpu static
    let cm = KernelAbstractions.CompilerMetadata{StaticSize{(64,)}, StaticSize{(128,)}, false}(nothing)
        @test KernelAbstractions.__ndrange(cm) isa CartesianIndices
        @test KernelAbstractions.__ndrange(cm) == CartesianIndices((128,))
        @test KernelAbstractions.__groupsize(cm) == 64
        @test KernelAbstractions.__groupindex(cm) === nothing
        @test KernelAbstractions.__dynamic_checkbounds(cm) == false
    end

    # cpu dynamic ndrange
    let cm = KernelAbstractions.CompilerMetadata{StaticSize{(64,)}, DynamicSize, false}(1, (128,), nothing)
        @test KernelAbstractions.__ndrange(cm) isa CartesianIndices
        @test KernelAbstractions.__ndrange(cm) == CartesianIndices((128,))
        @test KernelAbstractions.__groupsize(cm) == 64
        @test KernelAbstractions.__groupindex(cm) === 1
        @test KernelAbstractions.__dynamic_checkbounds(cm) == false
    end

    # gpu dynamic ndrange
    let cm = KernelAbstractions.CompilerMetadata{StaticSize{(64,)}, DynamicSize, false}((128,))
        @test KernelAbstractions.__ndrange(cm) isa CartesianIndices
        @test KernelAbstractions.__ndrange(cm) == CartesianIndices((128,))
        @test KernelAbstractions.__groupsize(cm) == 64
        @test KernelAbstractions.__groupindex(cm) === nothing
        @test KernelAbstractions.__dynamic_checkbounds(cm) == false
    end
end

identity(x)=x
@testset "partition" begin
    let kernel = KernelAbstractions.Kernel{CPU, StaticSize{(64,)}, DynamicSize, typeof(identity)}(identity)
        nworkgroups, dynamic = KernelAbstractions.partition(kernel, 128, nothing)
        @test nworkgroups == 2
        @test !dynamic

        nworkgroups, dynamic = KernelAbstractions.partition(kernel, 129, nothing)
        @test nworkgroups == 3
        @test dynamic

        nworkgroups, dynamic = KernelAbstractions.partition(kernel, 129, 64)
        @test nworkgroups == 3
        @test dynamic

        @test_throws ErrorException KernelAbstractions.partition(kernel,129, 65)
    end
    let kernel = KernelAbstractions.Kernel{CPU, StaticSize{(64,)}, StaticSize{(128,)}, typeof(identity)}(identity)
        nworkgroups, dynamic = KernelAbstractions.partition(kernel, 128, nothing)
        @test nworkgroups == 2
        @test !dynamic

        nworkgroups, dynamic = KernelAbstractions.partition(kernel, nothing, nothing)
        @test nworkgroups == 2
        @test !dynamic

        @test_throws ErrorException KernelAbstractions.partition(kernel, 129, nothing)
    end
end

@kernel function index_linear_global(A)
       I = @index(Global, Linear)
       A[I] = I
end
@kernel function index_linear_local(A)
       I  = @index(Global, Linear)
       li = @index(Local, Linear)
       A[I] = li
end
@kernel function index_cartesian_global(A)
       I = @index(Global, Cartesian)
       A[I] = I
end

function indextest(backend, ArrayT)
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
        ctx = KernelAbstractions.mkcontext(kernel, 1, nothing, nothing)
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
            ctx = KernelAbstractions.mkcontext(kernel, nothing)
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