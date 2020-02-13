using KernelAbstractions, CuArrays, Test, CUDAapi

@kernel function copy!(a,b)
    I = @index(Global)
    @inbounds b[I] = a[I]
end

@kernel function naive_transpose!(a, b)
  I = @index(Global, Cartesian)
  i, j = Tuple(I)
  @inbounds b[i, j] = a[j, i]
end

function main()

    # resolution of grid will be res*res
    res = 1024

    # creating initial arrays on CPU and GPU
    a = round.(rand(Float32, (res, res))*100)
    b = zeros(Float32, res, res)

    # beginning CPU tests
    println("CPU copy time is:")
    @time copy!(CPU(),4)(a, b, ndrange=size(a))

    println("CPU transpose time is:")
    @time naive_transpose!(ThreadedCPU(),4)(a, b, ndrange=size(a))

    println("Testing CPU transpose...")
    @test a == transpose(b)

    # beginning GPU tests
    if has_cuda_gpu()
        d_a = CuArray(a)
        d_b = CuArray(zeros(Float32, res, res))

        println("GPU copy time is:")
        CuArrays.@time copy!(CUDA(),1024)(d_a, d_b, ndrange=size(a))

        println("GPU transpose time is:")
        CuArrays.@time naive_transpose!(CUDA(),256)(d_a, d_b, ndrange=size(a))

        a = Array(d_a)
        b = Array(d_b)

        println("Testing GPU transpose...")
        @test a == transpose(b)
    end

    return nothing
end

main()

