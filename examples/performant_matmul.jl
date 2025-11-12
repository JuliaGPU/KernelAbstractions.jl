using KernelAbstractions
using StaticArrays
using Test
using Random
include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend

# We use a TILE_DIM of 16 as a safe value since while
#  most backends support up to 1024 threads per group,
#  Metal sometimes supports fewer.
const TILE_DIM = 16

function coalesced_matmul_kernel!(
        output, input1, input2, N, R, M,
        ::Val{TDIM}, ::Val{BANK} = Val(1)
    ) where {TDIM, BANK}
    gi, gj, _ = KI.get_group_id()
    i, j, _ = KI.get_local_id()

    # +1 to avoid bank conflicts on shared memory
    tile1 = KI.localmemory(eltype(output), (TDIM + BANK, TDIM))
    tile2 = KI.localmemory(eltype(output), (TDIM + BANK, TDIM))

    # variable for tile output
    outval = -zero(eltype(output))

    N = size(output, 1)
    # number of tiles depends on inner dimension
    NUM_TILES = div(R + TDIM - 1, TDIM)

    # loop over all tiles needed for this calculation
    for t in 0:(NUM_TILES - 1)
        # Can't use @index(Global), because we use a smaller ndrange
        I = (gi - 1) * TDIM + i
        J = (gj - 1) * TDIM + j

        # load inputs into tiles, with bounds checking for non-square matrices
        if I <= N && t * TDIM + j <= R
            @inbounds tile1[i, j] = input1[I, t * TDIM + j]
        else
            @inbounds tile1[i, j] = 0.0
        end
        if t * TILE_DIM + i <= R && J <= M
            @inbounds tile2[i, j] = input2[t * TDIM + i, J]
        else
            @inbounds tile2[i, j] = 0.0
        end

        # wait for all tiles to be loaded
        KI.barrier()

        # get global values again
        I = (gi - 1) * TDIM + i
        J = (gj - 1) * TDIM + j

        # calculate value of spot in output, use temporary value to allow for vectorization
        out = zero(eltype(output))
        @simd for k in 1:TDIM
            @inbounds out += tile1[i, k] * tile2[k, j]
        end
        outval += out

        KI.barrier()
    end

    # get global indices again
    I = (gi - 1) * TDIM + i
    J = (gj - 1) * TDIM + j

    # save if inbounds
    if I <= N && J <= M
        @inbounds output[I, J] = outval
    end
    return nothing
end

N = 1024
R = 512
M = 2048
A = rand!(allocate(backend, Float32, N, R))
B = rand!(allocate(backend, Float32, R, M))
C = KernelAbstractions.zeros(backend, Float32, N, M)

workgroupsize=(TILE_DIM, TILE_DIM)
numworkgroups=(cld(size(C, 1), TILE_DIM), cld(size(C, 2), TILE_DIM))

KI.@kernel backend workgroupsize numworkgroups coalesced_matmul_kernel!(C, A, B, N, R, M, Val(TILE_DIM))
KernelAbstractions.synchronize(backend)

@test isapprox(A * B, C)
