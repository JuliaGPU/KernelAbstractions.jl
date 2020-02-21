using KernelAbstractions
using Test
using CUDAapi
if has_cuda_gpu()
    using CuArrays
    CuArrays.allowscalar(false)
end

@kernel function localmem(A)
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    lmem = @localmem Int groupsize() # Ok iff groupsize is static 
    lmem[i] = i
    @synchronize
    A[I] = lmem[prod(groupsize()) - i + 1]
end

function harness(backend, ArrayT)
    A = ArrayT{Int}(undef, 64)
    wait(localmem(backend, 16)(A, ndrange=size(A)))
    @test all(A[1:16] .== 16:-1:1)
    @test all(A[17:32] .== 16:-1:1)
    @test all(A[33:48] .== 16:-1:1)
    @test all(A[49:64] .== 16:-1:1)
end

@testset "kernels" begin
    harness(CPU(), Array)
    if has_cuda_gpu()
        harness(CUDA(), CuArray)
    end
end
