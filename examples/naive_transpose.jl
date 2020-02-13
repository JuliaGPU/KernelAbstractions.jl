using KernelAbstractions, Test, CUDAapi
if CUDAapi.has_cuda_gpu()
    using CuArrays
    CuArrays.allowscalar(false)
end

@kernel function copy!(a,b)
    I = @index(Global)
    @inbounds b[I] = a[I]
end

@kernel function naive_transpose!(a, b)
  I = @index(Global, Cartesian)
  i, j = Tuple(I)
  @inbounds b[i, j] = a[j, i]
end

# creating wrapper functions
function launch_copy!(a, b)
    if size(a) != size(b)
        println("Matrix size mismatch!")
        return nothing
    end
    if isa(a, CuArray)
        kernel! = copy!(CUDA(),1024)
    else
        kernel! = copy!(CPU(),4)
    end
    kernel!(a, b, ndrange=size(a))
end

# creating wrapper functions
function launch_naive_transpose!(a, b)
    if size(a)[1] != size(b)[2] || size(a)[2] != size(b)[1]
        println("Matrix size mismatch!")
        return nothing
    end
    if isa(a, CuArray)
        kernel! = naive_transpose!(CUDA(),256)
    else
        kernel! = naive_transpose!(CPU(),4)
    end
    kernel!(a, b, ndrange=size(a))
end

function main()

    # resolution of grid will be res*res
    res = 1024

    # creating initial arrays on CPU and GPU
    a = round.(rand(Float32, (res, res))*100)
    b = zeros(Float32, res, res)

    # beginning CPU tests
    ev = launch_copy!(a, b)
    wait(ev)

    ev = launch_naive_transpose!(a,b)
    wait(ev)

    println("CPU transpose time is:")
    println("Testing CPU transpose...")
    @test a == transpose(b)

    # beginning GPU tests
    if has_cuda_gpu()
        d_a = CuArray(a)
        d_b = CuArray(zeros(Float32, res, res))

        launch_copy!(d_a, d_b)

        launch_naive_transpose!(d_a, d_b)

        a = Array(d_a)
        b = Array(d_b)

        println("Testing GPU transpose...")
        @test a == transpose(b)
    end

    return nothing
end

main()

