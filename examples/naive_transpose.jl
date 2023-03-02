using KernelAbstractions, Test, Random
include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend

@kernel function naive_transpose_kernel!(a, b)
    i, j = @index(Global, NTuple)
    @inbounds b[i, j] = a[j, i]
end

# create wrapper function to check inputs
# and select which backend to launch on.
function naive_transpose!(a, b)
    if size(a)[1] != size(b)[2] || size(a)[2] != size(b)[1]
        println("Matrix size mismatch!")
        return nothing
    end
    backend = get_backend(a)
    @assert get_backend(b) == backend
    groupsize = KernelAbstractions.is_gpu(backend) ? 256 : 1024
    kernel! = naive_transpose_kernel!(backend, groupsize)
    kernel!(a, b, ndrange=size(a))
end

# resolution of grid will be res*res
res = 1024

# creating initial arrays
a = rand!(allocate(backend, Float32, res, res))
b = KernelAbstractions.zeros(backend, Float32, res, res)

naive_transpose!(a,b)
KernelAbstractions.synchronize(backend)
@test a == transpose(b)

