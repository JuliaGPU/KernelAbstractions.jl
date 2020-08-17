using KernelAbstractions
using CUDA

has_cuda_gpu() || exit()

@kernel function transpose_kernel_naive!(b, a)
    i, j = @index(Global, NTuple)
    @inbounds b[i, j] = a[j, i]
end

const block_dim = 32
const grid_dim = 256

const T = Float32
const N = grid_dim * block_dim
const shape = N, N
const nreps = 1

NVTX.@range "Naive transpose ($block_dim, $block_dim)" let
    a = CuArray(rand(T, shape))
    b = similar(a, shape[2], shape[1])
    kernel! = transpose_kernel_naive!(CUDADevice(), (block_dim, block_dim), size(b))
  
    event = kernel!(b, a)
    wait(event)
    @assert Array(b) == Array(a)'
    CUDA.@profile begin
        for rep in 1:nreps
          event = kernel!(b, a, dependencies=(event,))
        end
        wait(event)
    end
end

NVTX.@range "Naive transpose ($(block_dim^2), 1)" let
    a = CuArray(rand(T, shape))
    b = similar(a, shape[2], shape[1])
    kernel! = transpose_kernel_naive!(CUDADevice(), (block_dim*block_dim, 1), size(b))
  
    event = kernel!(b, a)
    wait(event)
    @assert Array(b) == Array(a)'
    CUDA.@profile begin
        for rep in 1:nreps
          event = kernel!(b, a, dependencies=(event,))
        end
        wait(event)
    end
end

NVTX.@range "Naive transpose (1, $(block_dim^2))" let
    a = CuArray(rand(T, shape))
    b = similar(a, shape[2], shape[1])
    kernel! = transpose_kernel_naive!(CUDADevice(), (1, block_dim*block_dim), size(b))
  
    event = kernel!(b, a)
    wait(event)
    @assert Array(b) == Array(a)'
    CUDA.@profile begin
        for rep in 1:nreps
          event = kernel!(b, a, dependencies=(event,))
        end
        wait(event)
    end
end
