using KernelAbstractions
import SpecialFunctions


@kernel function gamma_knl(A, @Const(B))
    I = @index(Global)
    @inbounds A[I] = SpecialFunctions.gamma(B[I])
end

@kernel function erf_knl(A, @Const(B))
    I = @index(Global)
    @inbounds A[I] = SpecialFunctions.erf(B[I])
end

@kernel function erfc_knl(A, @Const(B))
    I = @index(Global)
    @inbounds A[I] = SpecialFunctions.erfc(B[I])
end

function specialfunctions_testsuite(Backend)
    backend = Backend()
    @testset "special functions: gamma" begin
        x = Float32[1.0, 2.0, 3.0, 5.5]

        cx = allocate(backend, Float32, length(x))
        KernelAbstractions.copyto!(backend, cx, x)
        cy = similar(cx)

        gamma_knl(backend)(cy, cx; ndrange = length(x))
        synchronize(backend)
        @test Array(cy) ≈ SpecialFunctions.gamma.(x)
    end

    @testset "special functions: erf" begin
        x = Float32[-1.0, -0.5, 0.0, 1.0e-3, 1.0, 2.0, 5.5]

        cx = allocate(backend, Float32, length(x))
        KernelAbstractions.copyto!(backend, cx, x)
        cy = similar(cx)

        erf_knl(backend)(cy, cx; ndrange = length(x))
        synchronize(backend)
        @test Array(cy) ≈ SpecialFunctions.erf.(x)
    end

    @testset "special functions: erfc" begin
        x = Float32[-1.0, -0.5, 0.0, 1.0e-3, 1.0, 2.0, 5.5]
        cx = allocate(backend, Float32, length(x))
        KernelAbstractions.copyto!(backend, cx, x)
        cy = similar(cx)

        erfc_knl(backend)(cy, cx; ndrange = length(x))
        synchronize(backend)
        @test Array(cy) ≈ SpecialFunctions.erfc.(x)
    end
    return
end
