# EXCLUDE FROM TESTING
if backend_str == "CUDA" && Base.find_package("CUDA") !== nothing
    using CUDA
    using CUDA.CUDAKernels
    const backend = CUDABackend()
    CUDA.allowscalar(false)
else
    const backend = CPU()
end

