using KernelAbstractions, Adapt, Test
include(joinpath(@__DIR__, "utils.jl")) # Load backend

if has_cuda && has_cuda_gpu()
    CUDA.allowscalar(false)
end

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
    device = get_computing_device(a)
    n = device isa GPU ? 256 : 4
    kernel! = naive_transpose_kernel!(device, n)
    kernel!(a, b, ndrange=size(a))
end


# resolution of grid will be res*res
res = 1024

# creating initial arrays
a = round.(rand(Float32, (res, res))*100)
b = zeros(Float32, res, res)

event = naive_transpose!(a,b)
wait(event)
@test a == transpose(b)

# beginning GPU tests
if has_cuda && has_cuda_gpu()
    d_a = CuArray(a)
    d_b = CUDA.zeros(Float32, res, res)

    ev = naive_transpose!(d_a, d_b)
    wait(ev)

    a = Array(d_a)
    b = Array(d_b)

    @test a == transpose(b)
end


if has_rocm && has_rocm_gpu()
    d_a = ROCArray(a)
    d_b = zeros(Float32, res, res) |> ROCArray

    ev = naive_transpose!(d_a, d_b)
    wait(ev)

    a = Array(d_a)
    b = Array(d_b)
    
    @test a == transpose(b)
end

