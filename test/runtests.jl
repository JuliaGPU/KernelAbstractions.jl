using KernelAbstractions
using Test

include("testsuite.jl")

@testset "CPU back-end" begin
    struct CPUBackendArray{T,N,A} end # Fake and unused
    Testsuite.testsuite(CPU, "CPU", Base, Array, CPUBackendArray)
end
