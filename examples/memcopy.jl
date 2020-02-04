using KernelAbstractions
using CUDAapi
using Test

@kernel function copy_kernel!(A, @Const(B))
    I = @index(Global)
    @inbounds A[I] = B[I]
end

A = zeros(128, 128)
B = ones(128, 128)

function mycopy!(A::Array, B::Array)
    @assert size(A) == size(B)
    kernel = copy_kernel!(ScalarCPU(), 32)
    kernel(A, B, ndrange=length(A))
end

function mycopy_static!(A::Array, B::Array)
    @assert size(A) == size(B)
    kernel = copy_kernel!(ScalarCPU(), 32, size(A)) # if size(A) varies this will cause recompilation
    kernel(A, B, ndrange=size(A))
end

event = mycopy!(A, B)
wait(event)
@test A == B

A = zeros(128, 128)
event = mycopy_static!(A, B)
wait(event)
@test A == B

function mycopy!(A::Array, B::Array)
    @assert size(A) == size(B)
    kernel = copy_kernel!(ThreadedCPU(), 8)
    kernel(A, B, ndrange=length(A))
end

function mycopy_static!(A::Array, B::Array)
    @assert size(A) == size(B)
    kernel = copy_kernel!(ThreadedCPU(), 32, size(A)) # if size(A) varies this will cause recompilation
    kernel(A, B, ndrange=size(A))
end

A = zeros(128, 128)
event = mycopy!(A, B)
wait(event)
@test A == B

A = zeros(128, 128)
event = mycopy_static!(A, B)
wait(event)
@test A == B

if has_cuda_gpu()
    using CuArrays

    function mycopy!(A::CuArray, B::CuArray)
        @assert size(A) == size(B)
        copy_kernel!(CUDA(), 256)(A, B, ndrange=length(A))
    end

    function mycopy_static!(A::CuArray, B::CuArray)
        @assert size(A) == size(B)
        kernel = copy_kernel!(CUDA(), 32, size(A)) # if size(A) varies this will cause recompilation
        kernel(A, B, ndrange=size(A))
    end

    A = CuArray{Float32}(undef, 1024)
    B = CuArrays.cuones(Float32, 1024)
    event = mycopy!(data)
    wait(event)
    @test A == B

    A = CuArray{Float32}(undef, 1024)
    event = mycopy_static!(data)
    wait(event)
    @test A == B
end
