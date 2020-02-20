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

@kernel function transpose_kernel!(b, a)
    block_dim_x, block_dim_y = block_dim, block_dim
    grid_dim_x, grid_dim_y = grid_dim, grid_dim

    wgsize = prod(groupsize())
    
    I = @index(Global)
    L = @index(Local)
    G = div(I - 1, wgsize) + 1
  
    thread_idx_x = (L - 1) % block_dim_x + 1
    thread_idx_y = div(L - 1, block_dim_x) + 1
    
    block_idx_x = (G - 1) % grid_dim_x + 1
    block_idx_y = div(G - 1, grid_dim_x) + 1
  
    i = (block_idx_x - 1) * block_dim_x + thread_idx_x
    j = (block_idx_y - 1) * block_dim_y + thread_idx_y
  
    @inbounds b[i + size(b, 1) * (j - 1)] = a[j + size(a, 1) * (i - 1)]
end

const T = Float32
const N = grid_dim * block_dim
const shape = N, N
const nreps = 10

NVTX.@range "Naive transpose $block_dim, $block_dim" let
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

NVTX.@range "Naive transpose $(block_dim^2), 1" let
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

NVTX.@range "Naive transpose 1, $(block_dim^2)" let
    a = CuArray(rand(T, shape))
    b = similar(a, shape[2], shape[1])
    kernel! = transpose_kernel_naive!(CUDA(), (1, blockdim*block_dim), size(b))
  
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

NVTX.@range "Baseline transpose" let
    a = CuArray(rand(T, shape))
    b = similar(a, shape[2], shape[1])
    
    kernel! = transpose_kernel!(CUDA(), (block_dim*block_dim), length(b))
  
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

