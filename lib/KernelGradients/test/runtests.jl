using KernelGradients
using KernelAbstractions
using Test
using Enzyme

include("testsuite.jl")

struct CPUDeviceArray{T,N,A} end # Fake and unused
GradientsTestsuite.testsuite(CPU, "CPU", Base, Array, CPUDeviceArray)
