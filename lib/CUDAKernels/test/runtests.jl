using KernelAbstractions
using KernelGradients
using Enzyme
using CUDA
using CUDAKernels
using Test
using SparseArrays

include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))
include(joinpath(dirname(pathof(KernelGradients)), "..", "test", "testsuite.jl"))

CI = parse(Bool, get(ENV, "CI", "false"))
if CI
    default = "CPU"
else
    default = "CUDA"
end

backend = get(ENV, "KERNELABSTRACTIONS_TEST_BACKEND", default)

if backend != "CUDA"
    @info "CUDA backend not selected"
    exit()
end

# if CUDA.functional(true)
if CUDA.functional()
    CUDA.versioninfo()
    CUDA.allowscalar(false)
    Testsuite.testsuite(CUDADevice{false}, backend, CUDA, CuArray, CUDA.CuDeviceArray)
    Testsuite.testsuite(CUDADevice{true}, backend, CUDA, CuArray, CUDA.CuDeviceArray)
    # GradientsTestsuite.testsuite(CUDADevice, backend, CUDA, CuArray, CUDA.CuDeviceArray)
elseif !CI
    error("No CUDA GPUs available!")
end