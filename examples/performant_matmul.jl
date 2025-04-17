using KernelAbstractions
using StaticArrays
using Test
using Random
include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend

const TILE_DIM = 32

@kernel function coalesced_matmul_kernel!(
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
    @inbounds outval[1] = zero(eltype(output))

    # number of tiles depends on inner dimension
    @uniform NUM_TILES = cld(R, TILE_DIM)

    # Can't use @index(Global), because we use a smaller ndrange
    I = (gi - 1) * TILE_DIM + i
    J = (gj - 1) * TILE_DIM + j
    # loop over all tiles needed for this calculation
    for t in 0:(NUM_TILES - 1)
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
        @synchronize(true)

        # calculate value of spot in output, use temporary value to allow for vectorization
        out = zero(eltype(output))
        @simd for k in 1:TILE_DIM
            @inbounds out += tile1[i, k] * tile2[k, j]
        end
        outval[1] += out

        @synchronize(true)
    end

    # save if inbounds
    if I <= N && J <= M
        @inbounds output[I, J] = outval[1]
    end
end

@testset "dims for $N, $R, $M" for (N,R,M) in [rand(500:1000,3) for _ in 1:10]
    A = rand!(allocate(backend, Float32, N, R))
    B = rand!(allocate(backend, Float32, R, M))
    C = KernelAbstractions.zeros(backend, Float32, N, M)
    
    kern = coalesced_matmul_kernel!(backend, (TILE_DIM, TILE_DIM))
    
    group_size_x = cld(N, TILE_DIM)
    group_size_y = cld(M, TILE_DIM)
    
    kern(C, A, B, N, R, M, ndrange = (group_size_x * TILE_DIM, group_size_y * TILE_DIM))
    KernelAbstractions.synchronize(backend)
    
    @test isapprox(A * B, C)
end
