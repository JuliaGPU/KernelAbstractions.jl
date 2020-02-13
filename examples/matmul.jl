using KernelAbstractions, Test, CUDAapi
if CUDAapi.has_cuda_gpu()
    using CuArrays
    CuArrays.allowscalar(false)
end

# Simple kernel for matrix multiplication
@kernel function matmul!(a, b, c)
    if size(a)[2] != size(b)[1]
        # here, we need a CPU / GPU generic print statement, like...
        # CUDAnative.@cuprintf("Matrix size mismatch!")
        return nothing
    end
    cI = @index(Global, Cartesian)

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = 0

    for i = 1:size(a)[2]
        tmp_sum += a[cI[1],i] * b[i,cI[2]]
    end

    c[cI] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function launch_matmul!(a, b, c)
    if size(a)[2] != size(b)[1]
        println("Matrix size mismatch!")
        return nothing
    end
    if isa(a, Array)
        kernel! = matmul!(CPU(),4)
    else
        kernel! = matmul!(CUDA(),256)
    end
    kernel!(a, b, c, ndrange=size(c)) 
end

function check()
    a = rand(256,123)
    b = rand(123, 45)
    c = zeros(256, 45)

    # beginning CPU tests, returns event
    ev = launch_matmul!(a,b,c)
    wait(ev)

    println("Testing CPU matrix multiplication...")
    @test isapprox(a*b, c)

    # beginning GPU tests
    if has_cuda_gpu()
        d_a = CuArray(a)
        d_b = CuArray(b)
        d_c = CuArray(c)

        ev = launch_matmul!(d_a, d_b, d_c)
        wait(ev)
        c = a*b

        println("Testing GPU matrix multiplication...")
        @test isapprox(Array(d_c), c)
    end
end

check()
