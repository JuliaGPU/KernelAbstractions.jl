using Test
using Enzyme
using KernelAbstractions

@kernel function square!(A)
    I = @index(Global, Linear)
    @inbounds A[I] *= A[I]
end

function caller(A, backend)
    kernel = square!(backend)
    kernel(A, ndrange=size(A))
    synchronize(backend)
end

function enzyme_testsuite(backend, ArrayT, supports_reverse=true)
    @testset "kernels" begin
        A = ArrayT{Float64}(undef, 64)
        A .= (1:1:64)
        dA = ArrayT{Float64}(undef, 64)
        dA .= 1

        if supports_reverse
            Enzyme.autodiff(Reverse, caller, Duplicated(A, dA), Const(backend()))
            @test all(dA .≈ (2:2:128))
        end

        A .= (1:1:64)
        dA .= 1

        Enzyme.autodiff(Forward, caller, Duplicated(A, dA), Const(backend()))
        @test all(dA .≈ 2:2:128)

    end
end
