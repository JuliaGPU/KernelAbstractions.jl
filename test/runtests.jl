using KernelAbstractions
using Test

include("testsuite.jl")

@testset "CPU back-end" begin
    struct CPUBackendArray{T,N,A} end # Fake and unused
    Testsuite.testsuite(CPU, "CPU", Base, Array, CPUBackendArray)
end

@kernel function kern_static(A)
    I = @index(Global)
    A[I] = Threads.threadid()
end

A = zeros(Int, Threads.nthreads())
kern_static(CPU(static=true), (1,))(A, ndrange=length(A))
@test A == 1:Threads.nthreads()

@kernel cpu=false function my_no_cpu_kernel(a)
end
@test_throws ErrorException("This kernel is unavailable for backend CPU") my_no_cpu_kernel(CPU())

struct NewBackend <: KernelAbstractions.GPU end
@testset "Default host implementation" begin
    backend = NewBackend()
    @test KernelAbstractions.isgpu(backend) == true

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

    kernel = my_no_cpu_kernel(backend)
    @test_throws MethodError kernel()
end

include("extensions/enzyme.jl")
@static if VERSION >= v"1.7.0"
    @testset "Enzyme" begin
        enzyme_testsuite(CPU, Array)
    end
end
