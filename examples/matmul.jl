using KernelAbstractions, Test, CUDAapi
if CUDAapi.has_cuda_gpu()
    using CuArrays
    CuArrays.allowscalar(false)
end

# Simple kernel for matrix multiplication
@kernel function matmul_kernel!(a, b, c)
    cI = @index(Global, Cartesian)

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = zero(eltype(c))
    for i = 1:size(a)[2]
        tmp_sum += a[cI[1],i] * b[i,cI[2]]
    end

    c[cI] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function matmul!(a, b, c)
    if size(a)[2] != size(b)[1]
        println("Matrix size mismatch!")
        return nothing
    end
    if isa(a, Array)
        kernel! = matmul_kernel!(CPU(),4)
    else
        kernel! = matmul_kernel!(CUDA(),256)
    end
    kernel!(a, b, c, ndrange=size(c)) 
end

a = rand(256,123)
b = rand(123, 45)
c = zeros(256, 45)

# beginning CPU tests, returns event
ev = matmul!(a,b,c)
wait(ev)

@test isapprox(c, a*b)

# beginning GPU tests
if has_cuda_gpu()
    d_a = CuArray(a)
    d_b = CuArray(b)
    d_c = CuArray(c)

    ev = matmul!(d_a, d_b, d_c)
    wait(ev)

    @test isapprox(Array(d_c), a*b)
end
