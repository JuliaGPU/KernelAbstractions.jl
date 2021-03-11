using KernelAbstractions
using Test

include("testsuite.jl")

backend = get(ENV, "KERNELABSTRACTIONS_TEST_BACKEND", "CPU")
if backend == "CPU"
    struct CPUDeviceArray{T,N,A} end # Fake and unused
    Testsuite.testsuite(CPU, backend, Base, Array, CPUDeviceArray)
elseif backend == "CUDA"
    using CUDAKernels, CUDA
    CUDA.versioninfo()
    if CUDA.functional(true)
        CUDA.allowscalar(false)
        Testsuite.testsuite(CUDADevice, backend, CUDA, CuArray, CUDA.CuDeviceArray)
    else
        error("No CUDA GPUs available!")
    end
else
    error("Unknown backend $backend")
end
