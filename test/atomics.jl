using KernelAbstractions
using KernelAbstractions: @atomic
using Test

@kernel function atomic_add_kernel!(hist)
    i = @index(Global, Linear)
    j = (i - 1) % length(hist) + 1
    @inbounds @atomic hist[j] += one(eltype(hist))
end

@kernel function atomic_max_kernel!(A)
    i = @index(Global, Linear)
    @inbounds @atomic max(A[1], eltype(A)(i))
end

@kernel function atomic_min_kernel!(A)
    i = @index(Global, Linear)
    @inbounds @atomic min(A[1], eltype(A)(i))
end

function atomics_testsuite(backend, ArrayT)
    if !KernelAbstractions.supports_atomics(backend())
        @test_skip "Backend does not support atomics"
        return
    end

    inttypes = [Int32, UInt32]
    eltypes = [inttypes; Float32]
    KernelAbstractions.supports_float64(backend()) && push!(eltypes, Float64)

    @testset "atomic add ($T)" for T in eltypes
        hist = ArrayT(zeros(T, 32))
        atomic_add_kernel!(backend())(hist; ndrange = 1024)
        synchronize(backend())
        @test all(Array(hist) .== T(1024 ÷ 32))
    end

    # Atomic min/max is only portable for integers: the CUDA backend has no native
    # floating-point min/max atomics and Atomix does not fall back to a CAS loop there.
    @testset "atomic max/min ($T)" for T in inttypes
        A = ArrayT(zeros(T, 1))
        atomic_max_kernel!(backend())(A; ndrange = 1024)
        synchronize(backend())
        @test Array(A)[1] == T(1024)

        A = ArrayT(fill(typemax(T), 1))
        atomic_min_kernel!(backend())(A; ndrange = 1024)
        synchronize(backend())
        @test Array(A)[1] == T(1)
    end
    return
end
