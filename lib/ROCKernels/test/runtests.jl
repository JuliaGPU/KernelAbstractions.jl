using KernelAbstractions
using KernelGradients
using Enzyme
using ROCKernels
using AMDGPU
using Test

include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))
include(joinpath(dirname(pathof(KernelGradients)), "..", "test", "testsuite.jl"))

if parse(Bool, get(ENV, "CI", "false"))
    default = "CPU"
else
    default = "ROCM"
end

backend = get(ENV, "KERNELABSTRACTIONS_TEST_BACKEND", default)

if backend != "ROCM"
    @info "ROCM backend not selected"
    exit()
end

if length(AMDGPU.get_agents(:gpu)) > 0
    AMDGPU.allowscalar(false)
    Testsuite.testsuite(ROCDevice, backend, AMDGPU, ROCArray, AMDGPU.ROCDeviceArray)
    GradientsTestsuite.testsuite(ROCDevice, backend, AMDGPU, ROCArray, AMDGPU.ROCDeviceArray)
else
    error("No AMD GPUs available!")
end