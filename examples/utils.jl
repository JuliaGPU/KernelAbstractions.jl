# EXCLUDE FROM TESTING

if !(@isdefined backend)
    const backend = if Base.find_package("CUDA") !== nothing
        using CUDA
        using CUDA.CUDAKernels
        CUDA.allowscalar(false)
        CUDABackend()
    else
        CPU()
    end
end
