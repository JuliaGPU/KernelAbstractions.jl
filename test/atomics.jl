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

@kernel function atomix_ordered!(A, B)
    i = @index(Global, Linear)
    T = eltype(A)
    @inbounds begin
        @atomic :release A[i] = T(i)
        v = @atomic :acquire A[i]
        @atomic :monotonic A[i] += one(T)
        @atomic :acquire_release A[i] += one(T)
        @atomic :sequentially_consistent A[i] += one(T)
        B[i] = @atomicswap :acquire_release A[i] = v + T(3)
    end
end

# UnsafeAtomics based kernels, operating on raw pointers

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

@kernel function unsafe_atomics_add_ordered!(hist, ordering)
    i = @index(Global, Linear)
    j = (i - 1) % length(hist) + 1
    UnsafeAtomics.add!(pointer(hist, j), one(eltype(hist)), ordering)
end

@kernel function unsafe_atomics_load_store_ordered!(A, B)
    i = @index(Global, Linear)
    v = UnsafeAtomics.load(pointer(B, i), UnsafeAtomics.acquire)
    UnsafeAtomics.store!(pointer(A, i), v, UnsafeAtomics.release)
end

# Non-blocking message passing: workitem 1 publishes data guarded by a flag
# with release/acquire fences; observers must see the data if they see the flag.
@kernel function unsafe_atomics_fence!(data, flag, observed)
    i = @index(Global, Linear)
    T = eltype(data)
    @inbounds if i == 1
        data[1] = T(42)
        UnsafeAtomics.fence(UnsafeAtomics.release)
        UnsafeAtomics.store!(pointer(flag, 1), one(T), UnsafeAtomics.monotonic)
    else
        f = UnsafeAtomics.load(pointer(flag, 1), UnsafeAtomics.monotonic)
        UnsafeAtomics.fence(UnsafeAtomics.acquire)
        observed[i] = f == one(T) ? data[1] : T(-1)
    end
end

@kernel function unsafe_atomics_syncscope!(A, hist)
    i = @index(Global, Linear)
    T = eltype(A)
    # contended, system scope
    j = (i - 1) % length(hist) + 1
    UnsafeAtomics.add!(pointer(hist, j), one(T), UnsafeAtomics.seq_cst, UnsafeAtomics.none)
    # uncontended, singlethread scope
    p = pointer(A, i)
    UnsafeAtomics.store!(p, T(i), UnsafeAtomics.monotonic, UnsafeAtomics.singlethread)
    UnsafeAtomics.fence(UnsafeAtomics.seq_cst, UnsafeAtomics.singlethread)
    UnsafeAtomics.add!(p, one(T), UnsafeAtomics.monotonic, UnsafeAtomics.singlethread)
end

function atomics_testsuite(backend, ArrayT)
    if !KernelAbstractions.supports_atomics(backend())
        @test_skip "Backend does not support atomics"
        return
    end

    @testset "Atomix" begin
        # Float32 is excluded since atomic float add requires the SPIR-V
        # extension SPV_EXT_shader_atomic_float_add, unavailable with PoCL.
        # TODO: use CAS-based fallbacks, cf. JuliaGPU/GPUCompiler.jl#652
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

        @testset "orderings" begin
            A = ArrayT(zeros(Int32, 256))
            B = ArrayT(zeros(Int32, 256))
            atomix_ordered!(backend())(A, B, ndrange = 256)
            synchronize(backend())
            @test Array(A) == (1:256) .+ 3
            @test Array(B) == (1:256) .+ 3
        end
    end

    @testset "UnsafeAtomics" begin
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

        @testset "ordering $ordering" for ordering in (
                UnsafeAtomics.monotonic, UnsafeAtomics.acquire, UnsafeAtomics.release,
                UnsafeAtomics.acq_rel, UnsafeAtomics.seq_cst,
            )
            hist = ArrayT(zeros(Int32, 32))
            unsafe_atomics_add_ordered!(backend())(hist, ordering, ndrange = 1024)
            synchronize(backend())
            @test all(Array(hist) .== 32)
        end

        @testset "ordered load/store" begin
            A = ArrayT(zeros(Int32, 256))
            B = ArrayT(collect(Int32, 1:256))
            unsafe_atomics_load_store_ordered!(backend())(A, B, ndrange = 256)
            synchronize(backend())
            @test Array(A) == 1:256
        end

        @testset "fences" begin
            data = ArrayT(zeros(Int32, 1))
            flag = ArrayT(zeros(Int32, 1))
            observed = ArrayT(zeros(Int32, 1024))
            unsafe_atomics_fence!(backend())(data, flag, observed, ndrange = 1024)
            synchronize(backend())
            # observers either did not see the flag (-1) or must see the data
            @test all(x -> x == -1 || x == 42, Array(observed)[2:end])
        end

        @testset "syncscopes" begin
            A = ArrayT(zeros(Int32, 1024))
            hist = ArrayT(zeros(Int32, 32))
            unsafe_atomics_syncscope!(backend())(A, hist, ndrange = 1024)
            synchronize(backend())
            @test Array(A) == (1:1024) .+ 1
            @test all(Array(hist) .== 32)
        end
    end
    return
end
