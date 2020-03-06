using KernelAbstractions, Test, CUDAapi
if has_cuda_gpu()
    using CuArrays, CUDAdrv
    CuArrays.allowscalar(false)
end

function GPU_copy_test(M)

    A = CuArray(rand(Float64, M))
    B = CuArray(rand(Float64, M))

    a = Array{Float64}(undef, M)
    println("1")
    pin!(CUDA(), a)

    copyevent = async_copy!(CUDA(), a, B)
    println("7")
    wait(copyevent)
    println("8")
    copyevent = async_copy!(CUDA(), A, a, dependencies=copyevent)
    wait(copyevent)
    println("9")

    @test isapprox(a, Array(A))
    @test isapprox(a, Array(B))
end

function CPU_copy_test(M)
    A = Array(rand(Float64, M))
    B = Array(rand(Float64, M))

    a = Array{Float64}(undef, M)

    copyevent = async_copy!(CPU(), a, B)
    copyevent = async_copy!(CPU(), A, a, dependencies=copyevent)
    wait(copyevent)

    @test isapprox(a, A)
    @test isapprox(a, B)
end

M = 1024

if has_cuda_gpu()
    GPU_copy_test(M)
end
CPU_copy_test(M)
