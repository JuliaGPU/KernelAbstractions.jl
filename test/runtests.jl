using KernelAbstractions
using Test

include("testsuite.jl")

backend = get(ENV, "KERNELABSTRACTIONS_TEST_BACKEND", "CPU")

if backend == "CPU"
    struct CPUDeviceArray{T,N,A} end # Fake and unused
    Testsuite.testsuite(CPU, backend, Base, Array, CPUDeviceArray)
elseif backend = "CUDA"
    using CUDA
    if CUDA.functional()
    CUDA.versioninfo()
    CUDA.allowscalar(false)
    Testsuite.testsuite(CUDADevice{false, false}, backend, CUDA, CuArray, CUDA.CuDeviceArray)
    for (PreferBlocks, AlwaysInline) in Iterators.product((true, false), (true, false))
        Testsuite.unittest_testsuite(CUDADevice{true, false}, backend, CUDA, CuArray, CUDA.CuDeviceArray)
    end
end
