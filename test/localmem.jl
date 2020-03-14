using KernelAbstractions
using Test
using CUDAapi
if has_cuda_gpu()
    using CuArrays
    CuArrays.allowscalar(false)
end

@kernel function localmem(A)
    N = @uniform prod(groupsize())
    @uniform begin
        N2 = prod(groupsize())
    end
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    lmem = @localmem Int (N,) # Ok iff groupsize is static 
    @inbounds begin
        lmem[i] = i
        @synchronize
        A[I] = lmem[N2 - i + 1]
    end
end

@kernel function localmem2(A)
    N = @uniform prod(groupsize())
    @uniform begin
        N2 = prod(groupsize())
    end
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    lmem = @localmem Int (N,) # Ok iff groupsize is static 
    @inbounds begin
        lmem[i] = i + 3
        for j in 1:2
            lmem[i] -= j
            @synchronize
        end
        A[I] = lmem[N2 - i + 1]
    end
end

function harness(backend, ArrayT)
    @testset for kernel! in (localmem(backend, 16), localmem2(backend, 16))
        A = ArrayT{Int}(undef, 64)
        wait(kernel!(A, ndrange=size(A)))
        @test all(A[1:16] .== 16:-1:1)
        @test all(A[17:32] .== 16:-1:1)
        @test all(A[33:48] .== 16:-1:1)
        @test all(A[49:64] .== 16:-1:1)
    end
end

@testset "kernels" begin
    harness(CPU(), Array)
    if has_cuda_gpu()
        harness(CUDA(), CuArray)
    end
end
