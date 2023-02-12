using KernelAbstractions, Test
include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend

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
mycopy!(A, B)
KernelAbstractions.synchronize(KernelAbstractions.get_device(A))
@test A == B


if has_cuda && has_cuda_gpu()
    function mycopy!(A::CuArray, B::CuArray)
        @assert size(A) == size(B)
        copy_kernel!(CUDADevice(), 256)(A, B, ndrange=length(A))
    end

    A = CuArray{Float32}(undef, 1024)
    B = CUDA.ones(Float32, 1024)
    mycopy!(A, B)
    KernelAbstractions.synchronize(KernelAbstractions.get_device(A))
    @test A == B
end

if has_rocm && has_rocm_gpu()
    function mycopy!(A::ROCArray, B::ROCArray)
        @assert size(A) == size(B)
        copy_kernel!(ROCDevice(), 256)(A, B, ndrange=length(A))
    end

    A = zeros(Float32, 1024) |> ROCArray
    B = ones(Float32, 1024) |> ROCArray
    mycopy!(A, B)
    KernelAbstractions.synchronize(KernelAbstractions.get_device(A))
    @test A == B
end
