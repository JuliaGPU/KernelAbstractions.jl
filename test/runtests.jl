using KernelAbstractions
using Test

include("testsuite.jl")

backend = get(ENV, "KERNELABSTRACTIONS_TEST_BACKEND", "CPU")

if backend == "CPU"
    struct CPUBackendArray{T,N,A} end # Fake and unused
    Testsuite.testsuite(CPU, backend, Base, Array, CPUBackendArray)
elseif backend == "CUDA"
    using CUDA
    if CUDA.functional()
        CUDA.versioninfo()
        CUDA.allowscalar(false)
        Testsuite.testsuite(CUDABackend{false, false}, backend, CUDA, CuArray, CUDA.CuBackendArray)
        for (PreferBlocks, AlwaysInline) in Iterators.product((true, false), (true, false))
            Testsuite.unittest_testsuite(CUDABackend{true, false}, backend, CUDA, CUDA.CuBackendArray)
        end
    end
end
