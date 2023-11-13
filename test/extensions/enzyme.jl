using Test
using Enzyme
using KernelAbstractions

@kernel function square_kernel!(A)
    I = @index(Global, Linear)
    @inbounds A[I] *= A[I]
end

@kernel function square2d_kernel!(A)
    I,J = @index(Global, NTuple)
    @inbounds A[I,J] *= A[I,J]
end

function square!(A, backend)
    kernel = square_kernel!(backend)
    kernel(A, ndrange=size(A))
    KernelAbstractions.synchronize(backend)
end

function square2d!(A, backend)
    kernel = square2d_kernel!(backend)
    kernel(A, ndrange=size(A))
    KernelAbstractions.synchronize(backend)
end

function enzyme_testsuite(backend, ArrayT, supports_reverse=true)
    @testset "kernels" begin
        @testset "Linear global index" begin
            A = ArrayT{Float64}(undef, 64)
            A .= (1:1:64)
            dA = ArrayT{Float64}(undef, 64)
            dA .= 1

            if supports_reverse
                Enzyme.autodiff(Reverse, square!, Duplicated(A, dA), Const(backend()))
                @test all(dA .≈ (2:2:128))
            end

            A .= (1:1:64)
            dA .= 1

            Enzyme.autodiff(Forward, square!, Duplicated(A, dA), Const(backend()))
            @test all(dA .≈ 2:2:128)
        end
        @testset "NTuple global index" begin
            A = ArrayT{Float64}(undef, 64, 64)
            A .= (1:1:64)
            dA = ArrayT{Float64}(undef, 64, 64)
            dA .= 1

            if supports_reverse
                Enzyme.autodiff(Reverse, square2d!, Duplicated(A, dA), Const(backend()))
                @test all(dA .≈ (2:2:128))
            end

            A .= (1:1:64)
            dA .= 1

            Enzyme.autodiff(Forward, square2d!, Duplicated(A, dA), Const(backend()))
            @test all(dA .≈ (2:2:128))
        end
    end
end
