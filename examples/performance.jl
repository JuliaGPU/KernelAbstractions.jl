using KernelAbstractions
using CUDAapi

CUDAapi.has_cuda_gpu() || exit()

using CuArrays
using CUDAdrv
using CUDAnative
using CUDAnative.NVTX

@kernel function transpose_kernel_naive!(b, a)
    I = @index(Global, Cartesian)
    i, j = I.I 
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
    kernel! = transpose_kernel_naive!(CUDA(), (block_dim, block_dim), size(b))
  
    event = kernel!(b, a)
    wait(event)
    @assert Array(b) == Array(a)'
    @CUDAdrv.profile begin
        for rep in 1:nreps
          event = kernel!(b, a, dependencies=(event,))
        end
        wait(event)
    end
end

NVTX.@range "Naive transpose ($(block_dim^2), 1)" let
    a = CuArray(rand(T, shape))
    b = similar(a, shape[2], shape[1])
    kernel! = transpose_kernel_naive!(CUDA(), (block_dim*block_dim, 1), size(b))
  
    event = kernel!(b, a)
    wait(event)
    @assert Array(b) == Array(a)'
    @CUDAdrv.profile begin
        for rep in 1:nreps
          event = kernel!(b, a, dependencies=(event,))
        end
        wait(event)
    end
end

NVTX.@range "Naive transpose (1, $(block_dim^2))" let
    a = CuArray(rand(T, shape))
    b = similar(a, shape[2], shape[1])
    kernel! = transpose_kernel_naive!(CUDA(), (1, block_dim*block_dim), size(b))
  
    event = kernel!(b, a)
    wait(event)
    @assert Array(b) == Array(a)'
    @CUDAdrv.profile begin
        for rep in 1:nreps
          event = kernel!(b, a, dependencies=(event,))
        end
        wait(event)
    end
end
