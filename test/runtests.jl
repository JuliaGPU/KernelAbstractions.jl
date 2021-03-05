using KernelAbstractions
using Test

include("testsuite.jl")

backend = get(ENV, "KERNELABSTRACTIONS_TEST_BACKEND", "CPU")
if backend == "CPU"
    struct CPUDeviceArray{T,N,A} end # Fake and unused
    Testsuite.testsuite(CPU, Array, CPUDeviceArray)
elseif backend == "CUDA"
    using CUDAKernels, CUDA
    if has_cuda_gpu()
        CUDA.allowscalar(false)
        Testsuite.testsuite(CUDADevice, CuArray, CUDA.CuDeviceArray)
    else
        error("No CUDA GPUs available!")
    end
else
    error("Unknown backend $backend")
end
