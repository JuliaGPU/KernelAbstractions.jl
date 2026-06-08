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

@testset "profiling_range defaults" begin
    backend = NewBackend()

    @test KernelAbstractions.profiling_range_active(backend) == false
    @test KernelAbstractions.profiling_range_active(nothing) == false

    @test KernelAbstractions.profiling_range_start(backend, "label") === nothing
    @test KernelAbstractions.profiling_range_start(backend, "label"; domain = "X") === nothing

    @test KernelAbstractions.profiling_range_end(backend, nothing) === nothing

    @test (@profiling_range backend "label" 1 + 2) == 3
    @test (@profiling_range backend "label" domain = "Custom" 1 + 2) == 3

    @test_throws ErrorException (@profiling_range backend "label" error("boom"))
end

@testset "profiling_range IntelITT extension" begin
    using IntelITT

    cpu = CPU()

    @test KernelAbstractions.profiling_range_active(cpu) isa Bool

    m = which(KernelAbstractions.profiling_range_active, Tuple{CPU})
    @test parentmodule(m) === Base.get_extension(KernelAbstractions, :IntelITTExt)

    task = KernelAbstractions.profiling_range_start(cpu, "label")
    @test task !== nothing
    @test KernelAbstractions.profiling_range_end(cpu, task) === nothing

    ext = Base.get_extension(KernelAbstractions, :IntelITTExt)
    @test ext.domain_for("Trixi") === ext.domain_for("Trixi")

    @test (@profiling_range cpu "label" 1 + 2) == 3
    @test (@profiling_range cpu "vi" domain = "Trixi" 7) == 7
end


# include("extensions/enzyme.jl")
# @static if VERSION >= v"1.7.0"
#     @testset "Enzyme" begin
#         enzyme_testsuite(CPU, Array)
#     end
# end
