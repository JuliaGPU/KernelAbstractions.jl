using KernelAbstractions
using KernelGradients
using Enzyme
using Metal
using MetalKernels
using Test
using SparseArrays

include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))
include(joinpath(dirname(pathof(KernelGradients)), "..", "test", "testsuite.jl"))

CI = parse(Bool, get(ENV, "CI", "false"))
if CI
    default = "CPU"
else
    default = "Metal"
end

backend = get(ENV, "KERNELABSTRACTIONS_TEST_BACKEND", default)

if backend != "Metal"
    @info "Metal backend not selected"
    exit()
end

Metal.versioninfo()
Metal.allowscalar(false)
Testsuite.testsuite(MetalDevice, backend, Metal, MtlArray, Metal.MtlDeviceArray)