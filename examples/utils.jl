# EXCLUDE FROM TESTING
import KernelInterface as KI

if !(@isdefined backend)
    if Base.find_package("CUDA") !== nothing
        using CUDA
        using CUDA.CUDAKernels
        const backend = CUDABackend()
        CUDA.allowscalar(false)
    else
        const backend = CPU()
    end
end

const f_type = KI.supports_float64(backend) ? Float64 : Float32
