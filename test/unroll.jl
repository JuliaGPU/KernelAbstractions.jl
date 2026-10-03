using KernelAbstractions
using KernelAbstractions.Extras
using StaticArrays

@kernel function kernel_unroll!(a)
    @unroll for i in 1:5
        @inbounds a[i] = i
    end
end

@kernel function kernel_unroll!(a, ::Val{N}) where {N}
    let M = N + 5
        @unroll for i in 6:M
            @inbounds a[i - 5] = i
        end
        @synchronize
    end
end

# Check that nested `@unroll` doesn't throw a syntax error
@kernel function kernel_unroll2!(A)
    @uniform begin
        a = MVector{3, Float32}(1, 2, 3)
        b = MVector{3, Float32}(3, 2, 1)
        c = MMatrix{3, 3, Float32}(undef)
    end
    I = @index(Global)
    @inbounds for m in 1:3
        @unroll for j in 1:3
            @unroll for i in 1:3
                c[1, j] = m * a[1] * b[j]
            end
        end
        A[I] = c[1, 1]
        @synchronize(m % 2 == 0)
    end
end

# `generated=true` makes the kernel a generated function, so that the `where`
# parameter `N` can be interpolated into `@unroll $N`, which requires a literal.
@kernel generated = true function kernel_unroll_generated!(a, ::Val{N}) where {N}
    @unroll $N for i in 1:5
        @inbounds a[i] = i * $N
    end
end

# `generated=true` composes with the other body transformations: `@Const`,
# `@localmem`, `@synchronize` and `inbounds=true` all round-trip through the quote.
@kernel generated = true inbounds = true function kernel_generated_transforms!(a, @Const(b), ::Val{N}) where {N}
    tile = @localmem Float32 (N,)
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    @unroll $N for k in 1:N
        tile[k] = b[k] * $N
    end
    @synchronize
    a[I] = tile[i]
end

# Errors while generating are reported through the return type rather than being
# swallowed as `Any`; the message must carry the original error.
@kernel generated = true function kernel_generated_closure!(a)
    I = @index(Global)
    @inbounds a[I] = sum(x -> x, 1:$(2))
end

@kernel generated = true function kernel_generated_badinterp!(a)
    I = @index(Global)
    @inbounds a[I] = $(length(a))
end

function unroll_testsuite(backend, ArrayT)
    a = ArrayT(zeros(Float32, 5))
    kernel! = kernel_unroll!(backend(), 1, 1)
    kernel!(a)
    kernel!(a, Val(5))
    kernel2! = kernel_unroll2!(backend(), 1, 1)
    kernel2!(a)
    synchronize(backend())

    a = ArrayT(zeros(Float32, 5))
    kernel3! = kernel_unroll_generated!(backend(), 1, 1)
    kernel3!(a, Val(2))
    synchronize(backend())
    @test Array(a) == Float32[2, 4, 6, 8, 10]

    a = ArrayT(zeros(Float32, 4))
    b = ArrayT(Float32[1, 2, 3, 4])
    kernel4! = kernel_generated_transforms!(backend(), 4, 4)
    kernel4!(a, b, Val(4))
    synchronize(backend())
    @test Array(a) == Float32[4, 8, 12, 16]

    a = ArrayT(zeros(Float32, 2))
    err = try
        kernel_generated_closure!(backend(), 2)(a; ndrange = 2)
        synchronize(backend())
        nothing
    catch e
        sprint(showerror, e)
    end
    @test occursin("GeneratedKernelError", err)
    @test occursin("cannot contain a closure", err)

    err = try
        kernel_generated_badinterp!(backend(), 2)(a; ndrange = 2)
        synchronize(backend())
        nothing
    catch e
        sprint(showerror, e)
    end
    @test occursin("GeneratedKernelError", err)
    @test occursin("MethodError: no method matching length(::Type{", err)
    return
end
