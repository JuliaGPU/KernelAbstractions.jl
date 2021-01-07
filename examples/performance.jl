using KernelAbstractions, CUDAKernels, CUDA, Test
using KernelAbstractions.Extras: @unroll

has_cuda_gpu() || exit()
CUDA.allowscalar(false)

const nreps = 3
const N = 2048
const T = Float32

const TILE_DIM = 32
const BLOCK_ROWS = 8

# Simple variants

@kernel function simple_copy_kernel!(output, @Const(input))
  I, J = @index(Global, NTuple)
  @inbounds output[I, J] = input[I, J]
end

@kernel function simple_transpose_kernel!(output, @Const(input))
    I, J = @index(Global, NTuple)
    @inbounds output[I, J] = input[I, J]
end

# Local memory variants

@kernel function lmem_copy_kernel!(output, @Const(input), 
                                   ::Val{BANK}=Val(1)) where BANK
    I, J = @index(Global, NTuple)
    i, j = @index(Local,  NTuple)

    N = @uniform groupsize()[1]
    M = @uniform groupsize()[2]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(output) (N+BANK, M)

    @inbounds tile[i, j] = input[I, J]

    @synchronize

    @inbounds output[I, J] = tile[i, j]
end

@kernel function lmem_transpose_kernel!(output, @Const(input),
                                       ::Val{BANK}=Val(1)) where BANK
    gi, gj = @index(Group, NTuple)
    i, j = @index(Local,  NTuple)

    N = @uniform groupsize()[1]
    M = @uniform groupsize()[2]
    
    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(output) (N+BANK, M) 

    # Manually calculate global indexes
    # Later on we need to pivot the group index
    I = (gi-1) * N + i
    J = (gj-1) * M + j

    @inbounds tile[i, j] = input[I, J]

    @synchronize

    # Pivot the group index
    I = (gj-1) * M + i
    J = (gi-1) * N + j

    @inbounds output[I, J] = tile[j, i]
end

# Local Memory + process multiple elements per lane

@kernel function coalesced_copy_kernel!(output, @Const(input), 
                                        ::Val{BANK}=Val(1)) where BANK
    gi, gj = @index(Group, NTuple)
    i, j   = @index(Local, NTuple)

    TILE_DIM   = @uniform groupsize()[1]
    BLOCK_ROWS = @uniform groupsize()[2]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(output) (TILE_DIM+BANK, TILE_DIM)

    # Can't use @index(Global), because we use a smaller ndrange
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    @unroll for k in 0:BLOCK_ROWS:(TILE_DIM-1)
        @inbounds tile[i, j+k] = input[I, J+k]
    end

    @synchronize

    @unroll for k in 0:BLOCK_ROWS:(TILE_DIM-1)
        @inbounds output[I, J+k] = tile[i, j+k]
    end
end

@kernel function coalesced_transpose_kernel!(output, @Const(input), 
                                             ::Val{BANK}=Val(1)) where BANK
    gi, gj = @index(Group, NTuple)
    i, j   = @index(Local, NTuple)

    TILE_DIM   = @uniform groupsize()[1]
    BLOCK_ROWS = @uniform groupsize()[2]

    # +1 to avoid bank conflicts on shared memory
    tile = @localmem eltype(output) (TILE_DIM+BANK, TILE_DIM)

    # Can't use @index(Global), because we use a smaller ndrange
    I = (gi-1) * TILE_DIM + i
    J = (gj-1) * TILE_DIM + j

    @unroll for k in 0:BLOCK_ROWS:(TILE_DIM-1)
        @inbounds tile[i, j+k] = input[I, J+k]
    end

    @synchronize

    # Transpose block offsets
    I = (gj-1) * TILE_DIM + i
    J = (gi-1) * TILE_DIM + j

    @unroll for k in 0:BLOCK_ROWS:(TILE_DIM-1)
        @inbounds output[I, J+k] = tile[j+k, i]
    end
end

# Benchmark simple

for block_dims in ((TILE_DIM, TILE_DIM), (TILE_DIM*TILE_DIM, 1), (1, TILE_DIM*TILE_DIM))
    for (name, kernel) in ( 
                            ("copy",      simple_copy_kernel!(CUDADevice(), block_dims)),
                            ("transpose", simple_transpose_kernel!(CUDADevice(), block_dims)),
                          )
        NVTX.@range "Simple $name $block_dims" let
            input = CUDA.rand(T, (N, N))
            output = similar(input)

            # compile kernel
            ev = kernel(input, output, ndrange=size(output))
            CUDA.@profile begin
                for rep in 1:nreps
                  ev = kernel(input, output, ndrange=size(output), dependencies=(ev,))
                end
                wait(ev)
            end
        end
    end
end

# Benchmark localmem
for (name, kernel) in ( 
                        ("copy",      lmem_copy_kernel!(CUDADevice(), (TILE_DIM, TILE_DIM))),
                        ("transpose", lmem_transpose_kernel!(CUDADevice(), (TILE_DIM, TILE_DIM))),
                      )
    for bank in (true, false)
        NVTX.@range "Localmem $name ($TILE_DIM, $TILE_DIM) bank=$bank" let
            input = CUDA.rand(T, (N, N))
            output = similar(input)

            # compile kernel
            ev = kernel(input, output, Val(Int(bank)), ndrange=size(output))
            CUDA.@profile begin
                for rep in 1:nreps
                    ev = kernel(input, output, Val(Int(bank)), ndrange=size(output), dependencies=(ev,))
                end
                wait(ev)
            end
        end
    end
end

# Benchmark localmem + multiple elements per lane
for (name, kernel) in ( 
                        ("copy",      coalesced_copy_kernel!(CUDADevice(), (TILE_DIM, BLOCK_ROWS))),
                        ("transpose", coalesced_transpose_kernel!(CUDADevice(), (TILE_DIM, BLOCK_ROWS))),
                      )
    for bank in (true, false)
        NVTX.@range "Localmem + multiple elements $name ($TILE_DIM, $BLOCK_ROWS) bank=$bank" let
            input = CUDA.rand(T, (N, N))
            output = similar(input)

            # We want a number of blocks equivalent to (TILE_DIM, TILE_DIM)
            # but our blocks are (TILE_DIM, BLOCK_ROWS) so we need to remove
            # a factor from the size of the array otherwise we get to many blocks
            block_factor = div(TILE_DIM, BLOCK_ROWS)
            ndrange = (N, div(N, block_factor))

            # compile kernel
            ev = kernel(input, output, Val(Int(bank)), ndrange=ndrange)
            CUDA.@profile begin
                for rep in 1:nreps
                    ev = kernel(input, output, Val(Int(bank)), ndrange=ndrange, dependencies=(ev,))
                end
                wait(ev)
            end
        end
    end
end
