using KernelAbstractions
using CUDAapi
using CUDAnative

using Test
using CuArrays

# Simple kernel for matrix multiplication
@kernel function matmul!(a, b, c)
    if size(a)[2] != size(b)[1]
        CUDAnative.@cuprintf("Matrix size mismatch!")
        return nothing
    end
    cI = CartesianIndices(c)[@index(Global)]

    tmp = 0

    for i = 1:size(a)[2]
        tmp += a[cI[1],i] * b[i,cI[2]]
    end

    c[cI] = tmp
end

function check()
    a = rand(256,123)
    b = rand(123, 45)
    c = zeros(256, 45)

    d_a = CuArray(a)
    d_b = CuArray(b)
    d_c = CuArray(c)

    matmul!(CUDA(),256)(d_a, d_b, d_c, ndrange=size(c))
    c = a*b

    if isapprox(Array(d_c), c)
        println("nice job, man.")
    else
        println("What a loser!")
    end
end

check()
