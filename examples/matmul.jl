using KernelAbstractions, Test, Random
include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend

# Simple kernel for matrix multiplication
@kernel function matmul_kernel!(a, b, c)
    i, j = @index(Global, NTuple)

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = zero(eltype(c))
    for k = 1:size(a)[2]
        tmp_sum += a[i,k] * b[k, j]
    end

    c[i,j] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function matmul!(a, b, c)
    if size(a)[2] != size(b)[1]
        println("Matrix size mismatch!")
        return nothing
    end
    backend = KernelAbstractions.get_backend(a)
    kernel! = matmul_kernel!(backend)
    kernel!(a, b, c, ndrange=size(c)) 
end

a = rand!(allocate(backend, Float32, 256, 123))
a = rand!(allocate(backend, Float32, 123, 45))
c = KernelAbstractions.zeros(backend, Float32, 256, 45)

matmul!(a,b,c)
KernelAbstractions.synchronize(backend)

@test isapprox(c, a*b)
