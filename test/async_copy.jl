using KernelAbstractions, Test, CUDAapi
if has_cuda_gpu()
    using CuArrays
    CuArrays.allowscalar(false)
end

function GPU_copy_test(M)

    A = CuArray(rand(Float64, M))
    B = CuArray(rand(Float64, M))

    a = Array{Float64}(undef, M)
    pin!(a)

    len = length(A)

    copystream = CuStream(CUDAdrv.STREAM_NON_BLOCKING)
    copyevent = recordevent(copystream)
    copyevent = async_copy!(pointer(A), pointer(B), M, stream=copystream)

    wait(copyevent)

    copyevent = async_copy!(pointer(a), pointer(B), M,
                            stream=copystream)
    copyevent = async_copy!(pointer(A), pointer(a), M,
                            stream=copystream, dependencies=copyevent)

    @test isapprox(a, Array(A))
    @test isapprox(a, Array(B))
end

function CPU_copy_test(M)
    A = Array(rand(Float64, M))
    B = Array(rand(Float64, M))

    a = Array{Float64}(undef, M)

    len = length(A)

    copyevent = async_copy!(pointer(a), pointer(B), M)
    copyevent = async_copy!(pointer(A), pointer(a), M)

    @test isapprox(a, Array(A))
    @test isapprox(a, Array(B))
end

M = 1024

if has_cuda_gpu()
    GPU_copy_test(M)
end
CPU_copy_test(M)
