using KernelAbstractions
using ROCKernels
using AMDGPU
using Test

include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))

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
else
    error("No AMD GPUs available!")
end