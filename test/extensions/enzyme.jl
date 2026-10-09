using Test
using Enzyme
using KernelAbstractions

@kernel function square!(A)
    I = @index(Global, Linear)
    @inbounds A[I] *= A[I]
end

function square_caller(A, backend)
    kernel = square!(backend)
    kernel(A, ndrange = size(A))
    KernelAbstractions.synchronize(backend)
    return
end


@kernel function mul!(A, B)
    I = @index(Global, Linear)
    @inbounds A[I] *= B
end

function mul_caller(A, B, backend)
    kernel = mul!(backend)
    kernel(A, B, ndrange = size(A))
    KernelAbstractions.synchronize(backend)
    return
end

@kernel function scale_typed!(A, ::Type{T}) where {T}
    I = @index(Global, Linear)
    @inbounds A[I] *= T(2)
end

function scale_typed_caller(A, ::Type{T}, backend) where {T}
    kernel = scale_typed!(backend)
    kernel(A, T, ndrange = size(A))
    KernelAbstractions.synchronize(backend)
    return
end

function enzyme_testsuite(backend, ArrayT, supports_reverse = true)
    @testset "kernels" begin
        A = ArrayT{Float64}(undef, 64)
        dA = ArrayT{Float64}(undef, 64)

        if supports_reverse

            A .= (1:1:64)
            dA .= 1

            Enzyme.autodiff(Reverse, square_caller, Duplicated(A, dA), Const(backend()))
            @test all(dA .≈ (2:2:128))

            # active arguments not support for GPU kernels
            if backend == CPU
                A .= (1:1:64)
                dA .= 1

                _, dB, _ = Enzyme.autodiff(Reverse, mul_caller, Duplicated(A, dA), Active(1.2), Const(backend()))[1]

                @test all(dA .≈ 1.2)
                @test dB ≈ sum(1:1:64)
            end
        end

        A .= (1:1:64)
        dA .= 1

        Enzyme.autodiff(Forward, square_caller, Duplicated(A, dA), Const(backend()))
        KernelAbstractions.synchronize(backend())
        @test all(dA .≈ 2:2:128)

        # Type arguments of kernels
        A .= (1:1:64)
        dA .= 1

        Enzyme.autodiff(Forward, scale_typed_caller, Duplicated(A, dA), Const(Float64), Const(backend()))
        KernelAbstractions.synchronize(backend())
        @test all(dA .≈ 2)
        @test all(A .≈ 2:2:128)

        # The strong zero setting is passed on to the kernel: the tangent of `A[I] * B` with
        # `B = Inf` in direction zero is `0 * Inf = NaN` without it and zero with it
        A .= 1
        dA .= 0

        Enzyme.autodiff(Forward, mul_caller, Duplicated(A, dA), Const(Inf), Const(backend()))
        KernelAbstractions.synchronize(backend())
        @test all(isnan, dA)

        A .= 1
        dA .= 0

        Enzyme.autodiff(Enzyme.set_strong_zero(Forward), mul_caller, Duplicated(A, dA), Const(Inf), Const(backend()))
        KernelAbstractions.synchronize(backend())
        @test all(iszero, dA)

    end
    return
end
