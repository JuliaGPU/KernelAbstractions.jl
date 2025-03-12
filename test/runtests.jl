using KernelAbstractions
using Test

include("testsuite.jl")

@info "Configuration" pocl = KernelAbstractions.POCL.nanoOpenCL.pocl_jll.libpocl

@testset "CPU back-end" begin
    struct CPUBackendArray{T, N, A} end # Fake and unused
    Testsuite.testsuite(CPU, "CPU", Base, Array, CPUBackendArray)
end

struct NewBackend <: KernelAbstractions.GPU end
@testset "Default host implementation" begin
    backend = NewBackend()

    @test_throws MethodError KernelAbstractions.synchronize(backend)

    @test_throws MethodError KernelAbstractions.allocate(backend, Float32, 1)
    @test_throws MethodError KernelAbstractions.allocate(backend, Float32, (1,))
    @test_throws MethodError KernelAbstractions.allocate(backend, Float32, 1, 2)

    @test_throws MethodError KernelAbstractions.zeros(backend, Float32, 1)
    @test_throws MethodError KernelAbstractions.ones(backend, Float32, 1)

    @test KernelAbstractions.supports_atomics(backend) == true
    @test KernelAbstractions.supports_float64(backend) == true

    @test KernelAbstractions.priority!(backend, :high) === nothing
    @test KernelAbstractions.priority!(backend, :normal) === nothing
    @test KernelAbstractions.priority!(backend, :low) === nothing

    @test_throws ErrorException KernelAbstractions.priority!(backend, :middle)

    @test KernelAbstractions.functional(backend) === missing
end


# include("extensions/enzyme.jl")
# @static if VERSION >= v"1.7.0"
#     @testset "Enzyme" begin
#         enzyme_testsuite(CPU, Array)
#     end
# end
