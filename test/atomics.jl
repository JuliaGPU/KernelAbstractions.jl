using KernelAbstractions
using KernelAbstractions: @atomic, @atomicswap, @atomicreplace
using Test

import UnsafeAtomics

# Atomix based kernels

@kernel function atomix_add!(hist)
    i = @index(Global, Linear)
    j = (i - 1) % length(hist) + 1
    @inbounds @atomic hist[j] += one(eltype(hist))
end

@kernel function atomix_max!(A)
    i = @index(Global, Linear)
    @inbounds @atomic max(A[1], eltype(A)(i))
end

@kernel function atomix_min!(A)
    i = @index(Global, Linear)
    @inbounds @atomic min(A[1], eltype(A)(i))
end

@kernel function atomix_load_store!(A, B)
    i = @index(Global, Linear)
    @inbounds begin
        v = @atomic B[i]
        @atomic A[i] = v
    end
end

@kernel function atomix_swap!(A, B)
    i = @index(Global, Linear)
    @inbounds B[i] = @atomicswap A[i] = eltype(A)(i)
end

@kernel function atomix_replace!(A, success)
    i = @index(Global, Linear)
    T = eltype(A)
    @inbounds begin
        # CAS that must succeed, followed by one that must fail
        (_, ok1) = @atomicreplace A[i] zero(T) => T(i)
        (_, ok2) = @atomicreplace A[i] zero(T) => T(-1)
        success[i] = ok1 & !ok2
    end
end

# UnsafeAtomics based kernels (CPU only, operating on raw pointers)

@kernel function unsafe_atomics_add!(hist)
    i = @index(Global, Linear)
    j = (i - 1) % length(hist) + 1
    UnsafeAtomics.add!(pointer(hist, j), one(eltype(hist)))
end

@kernel function unsafe_atomics_minmax!(A)
    i = @index(Global, Linear)
    T = eltype(A)
    UnsafeAtomics.max!(pointer(A, 1), T(i))
    UnsafeAtomics.min!(pointer(A, 2), T(i))
end

@kernel function unsafe_atomics_ops!(A, B)
    i = @index(Global, Linear)
    T = eltype(A)
    p = pointer(A, i)
    UnsafeAtomics.store!(p, T(i))
    old, new = UnsafeAtomics.modify!(p, +, T(1))
    (; success) = UnsafeAtomics.cas!(p, new, T(2) * new)
    if success
        old = UnsafeAtomics.xchg!(p, old)
    end
    B[i] = UnsafeAtomics.load(p)
end

@kernel function unsafe_atomics_ordering!(hist)
    i = @index(Global, Linear)
    j = (i - 1) % length(hist) + 1
    UnsafeAtomics.add!(pointer(hist, j), one(eltype(hist)), UnsafeAtomics.seq_cst)
end

function atomics_testsuite(backend, ArrayT)
    if !KernelAbstractions.supports_atomics(backend())
        @test_skip "Backend does not support atomics"
        return
    end

    @testset "Atomix" begin
        # Float32 is excluded since atomic float add requires the SPIR-V
        # extension SPV_EXT_shader_atomic_float_add, unavailable with PoCL.
        @testset "atomic add ($T)" for T in (Int32, UInt32)
            hist = ArrayT(zeros(T, 32))
            atomix_add!(backend())(hist, ndrange = 1024)
            synchronize(backend())
            @test all(Array(hist) .== T(1024 ÷ 32))
        end

        @testset "atomic max/min" begin
            A = ArrayT(zeros(Int32, 1))
            atomix_max!(backend())(A, ndrange = 1024)
            synchronize(backend())
            @test Array(A)[1] == 1024

            A = ArrayT(fill(typemax(Int32), 1))
            atomix_min!(backend())(A, ndrange = 1024)
            synchronize(backend())
            @test Array(A)[1] == 1
        end

        @testset "atomic load/store" begin
            A = ArrayT(zeros(Int32, 256))
            B = ArrayT{Int32}(collect(Int32, 1:256))
            atomix_load_store!(backend())(A, B, ndrange = 256)
            synchronize(backend())
            @test Array(A) == 1:256
        end

        @testset "atomicswap" begin
            A = ArrayT(fill(Int32(-1), 256))
            B = ArrayT(zeros(Int32, 256))
            atomix_swap!(backend())(A, B, ndrange = 256)
            synchronize(backend())
            @test Array(A) == 1:256
            @test all(Array(B) .== -1)
        end

        @testset "atomicreplace" begin
            A = ArrayT(zeros(Int32, 256))
            success = ArrayT(zeros(Bool, 256))
            atomix_replace!(backend())(A, success, ndrange = 256)
            synchronize(backend())
            @test Array(A) == 1:256
            @test all(Array(success))
        end
    end

    @testset "UnsafeAtomics" begin
        if !(backend() isa CPU)
            @test_skip "UnsafeAtomics tests only run on the CPU backend"
            return
        end

        @testset "atomic add ($T)" for T in (Int32, UInt32)
            hist = ArrayT(zeros(T, 32))
            unsafe_atomics_add!(backend())(hist, ndrange = 1024)
            synchronize(backend())
            @test all(Array(hist) .== T(1024 ÷ 32))
        end

        @testset "atomic max/min" begin
            A = ArrayT(Int32[0, typemax(Int32)])
            unsafe_atomics_minmax!(backend())(A, ndrange = 1024)
            synchronize(backend())
            @test Array(A) == [1024, 1]
        end

        @testset "store/modify/cas/xchg/load" begin
            A = ArrayT(zeros(Int32, 256))
            B = ArrayT(zeros(Int32, 256))
            unsafe_atomics_ops!(backend())(A, B, ndrange = 256)
            synchronize(backend())
            # store i, modify + 1, cas to 2(i + 1), xchg back to i
            @test Array(A) == 1:256
            @test Array(B) == 1:256
        end

        @testset "explicit ordering" begin
            hist = ArrayT(zeros(Int32, 32))
            unsafe_atomics_ordering!(backend())(hist, ndrange = 1024)
            synchronize(backend())
            @test all(Array(hist) .== 32)
        end
    end
    return
end
