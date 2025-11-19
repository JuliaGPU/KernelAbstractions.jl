using Random

const n = 256

function apply_seed(seed)
    if seed === missing
        # should result in different numbers across launches
        Random.seed!()
        # XXX: this currently doesn't work, because of the definition in Base,
        #      `seed!(r::MersenneTwister=default_rng())`, which breaks overriding
        #      `default_rng` with a non-MersenneTwister RNG.
    elseif seed !== nothing
        # should result in the same numbers
        Random.seed!(seed)
    elseif seed === nothing
        # should result in different numbers across launches,
        # as determined by the seed set during module loading.
    end
end

function random_testsuite(backend)
    eltypes = [Float16, Float32, Float64, Int32, UInt32, Int64, UInt64, Bool, UInt16]

    @testset "rand($T), seed $seed" for T in eltypes, seed in (nothing, #=missing,=# 1234)
        # different kernel invocations should get different numbers
        @testset "across launches" begin
            @kernel function kernel(A::AbstractArray{T}, seed) where {T}
                apply_seed(seed)
                tid = @index(Global, Linear)
                @inbounds A[tid] = rand(T)
            end

            a = KernelAbstractions.zeros(backend(), T, n)
            b = KernelAbstractions.zeros(backend(), T, n)

            kernel(backend())(a, seed, ndrange=n, workgroupsize=n)
            KernelAbstractions.synchronize(backend())
            kernel(backend())(b, seed, ndrange=n, workgroupsize=n)
            KernelAbstractions.synchronize(backend())

            if seed === nothing || seed === missing
                @test Array(a) != Array(b)
            else
                @test Array(a) == Array(b)
            end
        end

        # multiple calls to rand should get different numbers
        @testset "across calls" begin
            @kernel function kernel(A::AbstractArray{T}, B::AbstractArray{T}, seed) where {T}
                apply_seed(seed)
                tid = @index(Global, Linear)
                @inbounds A[tid] = rand(T)
                @inbounds B[tid] = rand(T)
            end

            a = KernelAbstractions.zeros(backend(), T, n)
            b = KernelAbstractions.zeros(backend(), T, n)

            kernel(backend())(a, b, seed, ndrange=n, workgroupsize=n)
            KernelAbstractions.synchronize(backend())

            @test Array(a) != Array(b)
        end

        if T != Bool
            # different threads should get different numbers
            @testset "across threads, dim $active_dim" for active_dim in 1:6
                @kernel function kernel(A::AbstractArray{T}, seed) where {T}
                    apply_seed(seed)
                    lid = @index(Local, NTuple)
                    gid = @index(Group, NTuple)
                    id = lid[1] * lid[2] * lid[3] * gid[1] * gid[2] * gid[3]
                    if 1 <= id <= length(A)
                        @inbounds A[id] = rand(T)
                    end
                end

                tx, ty, tz, bx, by, bz = [dim == active_dim ? 3 : 1 for dim in 1:6]
                gx, gy, gz = tx*bx, ty*by, tz*bz
                a = KernelAbstractions.zeros(backend(), T, 3)

                kernel(backend())(a, seed, ndrange=(gx, gy, gz), workgroupsize=(tx, ty, tz))
                KernelAbstractions.synchronize(backend())

                # NOTE: we don't just generate two numbers and compare them, instead generating a
                #       couple more and checking they're not all the same, in order to avoid
                #       occasional collisions with lower-precision types (i.e., Float16).
                @test length(unique(Array(a))) > 1
            end
        end
    end

    @testset "basic randn($T), seed $seed" for T in filter(x -> x <: Base.IEEEFloat, eltypes), seed in (nothing, #=missing,=# 1234)
        @kernel function kernel(A::AbstractArray{T}, seed) where {T}
            apply_seed(seed)
            tid = @index(Global, Linear)
            @inbounds A[tid] = randn(T)
        end

        a = KernelAbstractions.zeros(backend(), T, n)
        b = KernelAbstractions.zeros(backend(), T, n)

        kernel(backend())(a, seed, ndrange=n, workgroupsize=n)
        KernelAbstractions.synchronize(backend())
        kernel(backend())(b, seed, ndrange=n, workgroupsize=n)
        KernelAbstractions.synchronize(backend())

        if seed === nothing || seed === missing
            @test Array(a) != Array(b)
        else
            @test Array(a) == Array(b)
        end
    end

    randexp_eltypes = filter(x -> x <: Base.IEEEFloat, eltypes)
    # Check if we're using POCL and if it supports Float16 for randexp
    # POCL doesn't support log1p for Float16
    if backend == CPU
        filter!(x -> x != Float16, randexp_eltypes)
    end

    @testset "basic randexp($T), seed $seed" for T in randexp_eltypes, seed in (nothing, #=missing,=# 1234)
        @kernel function kernel(A::AbstractArray{T}, seed) where {T}
            apply_seed(seed)
            tid = @index(Global, Linear)
            @inbounds A[tid] = randexp(T)
        end

        a = KernelAbstractions.zeros(backend(), T, n)
        b = KernelAbstractions.zeros(backend(), T, n)

        kernel(backend())(a, seed, ndrange=n, workgroupsize=n)
        KernelAbstractions.synchronize(backend())
        kernel(backend())(b, seed, ndrange=n, workgroupsize=n)
        KernelAbstractions.synchronize(backend())

        if seed === nothing || seed === missing
            @test Array(a) != Array(b)
        else
            @test Array(a) == Array(b)
        end
    end

    @testset "rand(::AbstractRange{$T}), seed $seed" for T in (Int32, Int64, UInt32, UInt64), seed in (nothing, #=missing,=# 1234)
        @kernel function kernel(A::AbstractArray{T}, seed) where {T}
            apply_seed(seed)
            tid = @index(Global, Linear)
            @inbounds A[tid] = rand(T(10):T(20))
        end

        a = KernelAbstractions.zeros(backend(), T, n)
        b = KernelAbstractions.zeros(backend(), T, n)

        kernel(backend())(a, seed, ndrange=n, workgroupsize=n)
        KernelAbstractions.synchronize(backend())
        kernel(backend())(b, seed, ndrange=n, workgroupsize=n)
        KernelAbstractions.synchronize(backend())

        if seed === nothing || seed === missing
            @test Array(a) != Array(b)
        else
            @test Array(a) == Array(b)
        end
    end
end
