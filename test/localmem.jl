using KernelAbstractions
using Test

@kernel function localmem(A)
    N = @uniform prod(@groupsize())
    @uniform begin
        N2 = prod(@groupsize())
    end
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    lmem = @localmem Int (N,) # Ok iff groupsize is static
    @inbounds begin
        lmem[i] = i
        @synchronize
        A[I] = lmem[N2 - i + 1]
    end
end

@kernel function localmem2(A)
    N = @uniform prod(@groupsize())
    @uniform begin
        N2 = prod(@groupsize())
    end
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    lmem = @localmem Int (N,) # Ok iff groupsize is static
    @inbounds begin
        lmem[i] = i + 3
        for j in 1:2
            lmem[i] -= j
            @synchronize
        end
        A[I] = lmem[N2 - i + 1]
    end
end

@kernel unsafe_indices = true function localmem_unsafe_indices(A)
    N = @uniform prod(@groupsize())
    gI = @index(Group, Linear)
    i = @index(Local, Linear)
    lmem = @localmem Int (N,) # Ok iff groupsize is static
    lmem[i] = i
    @synchronize
    I = (gI - 1) * N + i
    if I <= length(A)
        A[I] = lmem[N - i + 1]
    end
end

@kernel function many_localmem(A)
    N = @uniform prod(@groupsize())
    @uniform begin
        N2 = prod(@groupsize())
    end
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    lmem1 = @localmem Int (N,) # Ok iff groupsize is static
    lmem2 = @localmem Int (N,) # Ok iff groupsize is static
    @inbounds begin
        lmem1[i] = i - 1
        lmem2[i] = 1
        @synchronize
        A[I] = lmem1[N2 - i + 1] + lmem2[N2 - i + 1]
    end
end

# A kernel whose `@localmem` was expanded before `@kernel` saw it, as tools that generate
# kernels (e.g. ParallelStencil.jl) do. `@kernel` copies the definition with `deepcopy`,
# which fails for expansions that contain a module.
const pre_expanded_localmem = macroexpand(@__MODULE__, :(KernelAbstractions.@localmem Int (16,)))
@eval @kernel function localmem_pre_expanded(A)
    i = @index(Local, Linear)
    I = @index(Global, Linear)
    lmem = $pre_expanded_localmem
    lmem[i] = i
    @synchronize
    @inbounds A[I] = lmem[16 - i + 1]
end

function localmem_testsuite(backend, ArrayT)
    @testset "pre-expanded @localmem" begin
        @test deepcopy(pre_expanded_localmem) == pre_expanded_localmem
        A = ArrayT{Int}(undef, 32)
        localmem_pre_expanded(backend(), 16)(A, ndrange = size(A))
        synchronize(backend())
        @test Array(A) == [17 .- (1:16); 17 .- (1:16)]
    end

    @testset "kernels" begin
        @testset for kernel! in (localmem(backend(), 16), localmem2(backend(), 16), localmem_unsafe_indices(backend(), 16), many_localmem(backend(), 16))
            A = ArrayT{Int}(undef, 64)
            kernel!(A, ndrange = size(A))
            synchronize(backend())
            B = Array(A)
            @test all(B[1:16] .== 16:-1:1)
            @test all(B[17:32] .== 16:-1:1)
            @test all(B[33:48] .== 16:-1:1)
            @test all(B[49:64] .== 16:-1:1)
        end
    end
    return
end
