using KernelAbstractions
using CUDAKernels
using CUDA
using Test

@kernel function copy_kernel!(A, @Const(B))
    I = @index(Global)
    @inbounds A[I] = B[I]
end

function mycopy!(A::Array, B::Array)
    @assert size(A) == size(B)
    kernel = copy_kernel!(CPU(), 8)
    kernel(A, B, ndrange=length(A))
end

A = zeros(128, 128)
B = ones(128, 128)
event = mycopy!(A, B)
wait(event)
@test A == B


if has_cuda_gpu()

    function mycopy!(A::CuArray, B::CuArray)
        @assert size(A) == size(B)
        copy_kernel!(CUDADevice(), 256)(A, B, ndrange=length(A))
    end

    A = CuArray{Float32}(undef, 1024)
    B = CUDA.ones(Float32, 1024)
    event = mycopy!(A, B)
    wait(event)
    @test A == B
end
