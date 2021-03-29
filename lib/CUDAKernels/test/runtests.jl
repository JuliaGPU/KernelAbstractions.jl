using KernelAbstractions
using CUDA
using CUDAKernels
using Test

include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))

if parse(Bool, get(ENV, "CI", "false"))
    default = "CPU"
else
    default = "CUDA"
end

backend = get(ENV, "KERNELABSTRACTIONS_TEST_BACKEND", default)

if backend != "CUDA"
    @info "CUDA backend not selected"
    exit()
end

CUDA.versioninfo()
if CUDA.functional(true)
    CUDA.allowscalar(false)
    Testsuite.testsuite(CUDADevice, backend, CUDA, CuArray, CUDA.CuDeviceArray)
else
    error("No CUDA GPUs available!")
end