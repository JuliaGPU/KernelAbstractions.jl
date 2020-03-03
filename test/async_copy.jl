using KernelAbstractions, CuArrays, CUDAnative, CUDAdrv, Test

function main()
    T = Float32
    M = 1024
    N = 1024

    A = CuArray(rand(T, M, N))
    B = CuArray(rand(T, M, N))

    a = Array{T}(undef, M, N)
    #pin!(a)

    len = length(A)

    copystream = CuStream(CUDAdrv.STREAM_NON_BLOCKING)
    copyevent = recordevent(copystream)
    copyevent = async_copy!(pointer(A), pointer(B), M*N, stream=copystream)

    wait(copyevent)

    copyevent = async_copy!(pointer(B), pointer(a), M*N, stream=copystream)
    copyevent = async_copy!(pointer(a), pointer(A), M*N,
                            stream=copystream, dependencies=copyevent)

    @test isapprox(a, Array(A))
    @test isapprox(a, Array(B))
end

main()
