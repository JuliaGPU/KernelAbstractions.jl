using KernelAbstractions
using Test

include("testsuite.jl")

backend = get(ENV, "KERNELABSTRACTIONS_TEST_BACKEND", "CPU")

if backend != "CPU"
    @info "CPU backend not selected"
    exit()
end

struct CPUDeviceArray{T,N,A} end # Fake and unused
Testsuite.testsuite(CPU, backend, Base, Array, CPUDeviceArray)
