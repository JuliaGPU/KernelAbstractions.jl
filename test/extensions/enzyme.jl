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

function enzyme_testsuite(backend, ArrayT, supports_reverse = true)
    @testset "kernels" begin
        A = ArrayT{Float64}(undef, 64)
        dA = ArrayT{Float64}(undef, 64)

        if supports_reverse

            A .= (1:1:64)
            dA .= 1

            Enzyme.autodiff(Reverse, square_caller, Duplicated(A, dA), Const(backend()))
            @test all(dA .≈ (2:2:128))

            A .= (1:1:64)
            dA .= 1

            _, dB, _ = Enzyme.autodiff(Reverse, mul_caller, Duplicated(A, dA), Active(1.2), Const(backend()))[1]

            @test all(dA .≈ 1.2)
            @test dB ≈ sum(1:1:64)

            # Batched reverse with an `Active` argument: one shadow per lane.
            A .= (1:1:64)
            dA1 = similar(A)
            dA2 = similar(A)
            dA1 .= 1
            dA2 .= 2

            _, dBs, _ = Enzyme.autodiff(Reverse, mul_caller, BatchDuplicated(A, (dA1, dA2)), Active(1.2), Const(backend()))[1]

            @test dBs isa NTuple{2, Float64}
            @test dBs[1] ≈ sum(1:1:64)
            @test dBs[2] ≈ 2 * sum(1:1:64)
            @test all(dA1 .≈ 1.2)
            @test all(dA2 .≈ 2.4)
        end

        A .= (1:1:64)
        dA .= 1

        Enzyme.autodiff(Forward, square_caller, Duplicated(A, dA), Const(backend()))
        KernelAbstractions.synchronize(backend())
        @test all(dA .≈ 2:2:128)

    end
    return
end
