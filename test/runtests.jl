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
