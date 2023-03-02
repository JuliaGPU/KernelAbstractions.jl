# EXCLUDE FROM TESTING
if Base.find_package("CUDA") !== nothing
    using CUDA
    using CUDA.CUDAKernels
    const backend = CUDABackend()
else
    const backend = CPU()
end

