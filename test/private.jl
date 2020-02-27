using KernelAbstractions
using Test
using CUDAapi
if has_cuda_gpu()
    using CuArrays
    CuArrays.allowscalar(false)
end

@kernel function private(A)
    N = prod(groupsize())
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    priv = @private Int (1,)
    priv[1] = N - i + 1
    @synchronize
    A[I] = priv[1]
end

function harness(backend, ArrayT)
    A = ArrayT{Int}(undef, 64)
    wait(private(backend, 16)(A, ndrange=size(A)))
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
