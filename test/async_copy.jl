using KernelAbstractions, Test, CUDA

if has_cuda_gpu()
    CUDA.allowscalar(false)
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
    copy_test(CUDADevice(), CuArray, M)
end
copy_test(CPU(), Array, M)