using KernelAbstractions, Test, CUDAapi
if has_cuda_gpu()
    using CuArrays, CUDAdrv
    CuArrays.allowscalar(false)
end

function copy_test(backend, ArrayT, M)
    A = ArrayT(rand(Float64, M))
    B = ArrayT(rand(Float64, M))

    a = Array{Float64}(undef, M)
    event = async_copy!(backend, a, B, dependencies=Event(CPU()))
    event = async_copy!(backend, A, a, dependencies=event)
    wait(event)

    @test isapprox(a, Array(A))
    @test isapprox(a, Array(B))
end

M = 1024

if has_cuda_gpu()
    copy_test(CUDA(), CuArray, M)
end
copy_test(CPU(), Array, M)
