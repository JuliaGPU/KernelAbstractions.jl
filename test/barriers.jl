using KernelAbstractions
using Test

# These kernels are launched with an `ndrange` of 13 and workgroups of 8, so the last
# workgroup has 3 padding work-items. The arrays have room for those work-items, so
# that it can be checked that they don't execute guarded code.

# A `@uniform` statement after the last `@synchronize` in a loop body.
@kernel function trailing_uniform!(A)
    i = @index(Global)
    @uniform remaining = 3
    @uniform iterations = 0
    while remaining > 0 && iterations < 10
        @uniform iterations += 1
        A[i] += 1
        @synchronize()
        @uniform remaining -= 1
    end
end

# Branches without a `@synchronize` next to one with a `@synchronize`.
@kernel function branches!(A, flag)
    i = @index(Global)
    if flag == 1
        @synchronize()
        A[i] = 1
    elseif flag == 2
        A[i] = 2
    else
        A[i] = 3
    end
end

# Module-qualified uses of the kernel language.
@kernel function qualified!(A)
    i = @index(Global)
    li = @index(Local)
    KernelAbstractions.@uniform n = 2
    tile = KernelAbstractions.@localmem Int (8,)
    for _ in 1:n
        tile[li] = i
        KernelAbstractions.@synchronize()
        A[i] += tile[li]
        KernelAbstractions.@synchronize()
    end
end

function barrier_testsuite(backend, ArrayT)
    @testset "recognized barriers" begin
        @test KernelAbstractions.is_sync(:(@synchronize()))
        @test KernelAbstractions.is_sync(:(@synchronize))
        @test KernelAbstractions.is_sync(:(KernelAbstractions.@synchronize()))
        @test KernelAbstractions.is_sync(:(KA.@synchronize))
        @test !KernelAbstractions.is_sync(:(@uniform x = 1))
        @test !KernelAbstractions.is_sync(:(synchronize()))
    end

    @testset "trailing @uniform" begin
        A = KernelAbstractions.zeros(backend(), Int, 16)
        trailing_uniform!(backend(), 8)(A; ndrange = 13)
        @test Array(A) == [fill(3, 13); zeros(Int, 3)]
    end

    @testset "branches without @synchronize" begin
        for flag in 1:3
            A = KernelAbstractions.zeros(backend(), Int, 16)
            branches!(backend(), 8)(A, flag; ndrange = 13)
            @test Array(A) == [fill(flag, 13); zeros(Int, 3)]
        end
    end

    @testset "qualified macros" begin
        A = KernelAbstractions.zeros(backend(), Int, 16)
        qualified!(backend(), 8)(A; ndrange = 13)
        @test Array(A) == [2 .* (1:13); zeros(Int, 3)]
    end

    # padding work-items skip the code between `@synchronize`s, so they would read these
    # variables undefined (or never update them) in conditions that they do evaluate
    @testset "conditions reading masked variables" begin
        @test_throws "reads `n`" @macroexpand @kernel function f(A)
            n = length(A)
            for _ in 1:n
                @synchronize()
            end
        end
        @test_throws "reads `j`" @macroexpand @kernel function f(A)
            @uniform j = 0
            while j < 4
                @synchronize()
                j += 1
            end
        end
        @test_throws "reads `n`" @macroexpand @kernel function f(A)
            @inbounds n = A[1]
            while n > 0
                @synchronize()
                @uniform n -= 1
            end
        end
        @test_throws "reads `flag`" @macroexpand @kernel function f(A)
            flag = A[1] > 0
            if flag
                @synchronize()
            end
        end
        # without the implicit bounds check, there are no padding work-items
        @test @macroexpand(
            @kernel unsafe_indices = true function f(A)
                n = length(A)
                for _ in 1:n
                    @synchronize()
                end
            end
        ) isa Expr
        # neither keyword arguments, named tuples, closures nor loop variables assign `n`
        @test @macroexpand(
            @kernel function f(A, n)
                x = sum(A; init = 0)
                t = (n = 1,)
                g = n -> n + 1
                for n in 1:2
                end
                @uniform k = 0
                while k < n
                    @synchronize()
                    @uniform k += 1
                end
            end
        ) isa Expr
    end
    return
end
