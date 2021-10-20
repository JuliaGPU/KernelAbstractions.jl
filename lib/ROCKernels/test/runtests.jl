using KernelAbstractions
using KernelGradients
using Enzyme
using ROCKernels
using AMDGPU
using Test

include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))
include(joinpath(dirname(pathof(KernelGradients)), "..", "test", "testsuite.jl"))

CI = parse(Bool, get(ENV, "CI", "false"))
if CI
    default = "CPU"
else
    default = "ROCM"
end

backend = get(ENV, "KERNELABSTRACTIONS_TEST_BACKEND", default)

if backend != "ROCM"
    @info "ROCM backend not selected"
    exit()
end

if AMDGPU.has_rocm_gpu()
    AMDGPU.allowscalar(false)
    Testsuite.testsuite(ROCDevice, backend, AMDGPU, ROCArray, AMDGPU.ROCDeviceArray)
    # GradientsTestsuite.testsuite(ROCDevice, backend, AMDGPU, ROCArray, AMDGPU.ROCDeviceArray)
elseif !CI
    error("No AMD GPUs available!")
end

@test "get_device" begin
    @test @inferred(KernelAbstractions.get_device(ROCArray(rand(Float32, 3, 3)))) == ROCDevice()
end
