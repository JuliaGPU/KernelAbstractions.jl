using KernelAbstractions
using StaticArrays
using Test
using Random
include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend

# We use a TILE_DIM of 16 as a safe value since while
#  most backends support up to 1024 threads per group,
#  Metal sometimes supports fewer.
const TILE_DIM = 16

@kernel unsafe_indices = true function coalesced_matmul_kernel!(
        output, @Const(input1), @Const(input2), N, R, M,
        ::Val{BANK} = Val(1),
    ) where {BANK}
    gi, gj = @index(Group, NTuple)
    i, j = @index(Local, NTuple)

    TILE_DIM = @uniform @groupsize()[1]

    # +1 to avoid bank conflicts on shared memory
    tile1 = @localmem eltype(output) (TILE_DIM + BANK, TILE_DIM)
    tile2 = @localmem eltype(output) (TILE_DIM + BANK, TILE_DIM)

    # private variable for tile output
    outval = @private eltype(output) 1
    @inbounds outval[1] = -zero(eltype(output))

    @uniform N = size(output, 1)
    # number of tiles depends on inner dimension
    @uniform NUM_TILES = div(R + TILE_DIM - 1, TILE_DIM)

    # loop over all tiles needed for this calculation
    for t in 0:(NUM_TILES - 1)
        # Can't use @index(Global), because we use a smaller ndrange
        I = (gi - 1) * TILE_DIM + i
        J = (gj - 1) * TILE_DIM + j

        # load inputs into tiles, with bounds checking for non-square matrices
        if I <= N && t * TILE_DIM + j <= R
            @inbounds tile1[i, j] = input1[I, t * TILE_DIM + j]
        else
            @inbounds tile1[i, j] = 0.0
        end
        if t * TILE_DIM + i <= R && J <= M
            @inbounds tile2[i, j] = input2[t * TILE_DIM + i, J]
        else
            @inbounds tile2[i, j] = 0.0
        end

        # wait for all tiles to be loaded
        @synchronize

        # get global values again
        I = (gi - 1) * TILE_DIM + i
        J = (gj - 1) * TILE_DIM + j

        # calculate value of spot in output, use temporary value to allow for vectorization
        out = zero(eltype(output))
        @simd for k in 1:TILE_DIM
            @inbounds out += tile1[i, k] * tile2[k, j]
        end
        outval[1] += out

        @synchronize
    end

    # get global indices again
    I = (gi - 1) * TILE_DIM + i
    J = (gj - 1) * TILE_DIM + j

    # save if inbounds
    if I <= N && J <= M
        @inbounds output[I, J] = outval[1]
    end
end

N = 1024
R = 512
M = 2048
A = rand!(allocate(backend, Float32, N, R))
B = rand!(allocate(backend, Float32, R, M))
C = KernelAbstractions.zeros(backend, Float32, N, M)

kern = coalesced_matmul_kernel!(backend, (TILE_DIM, TILE_DIM))

kern(C, A, B, N, R, M, ndrange = size(C))
KernelAbstractions.synchronize(backend)

@test isapprox(A * B, C)
