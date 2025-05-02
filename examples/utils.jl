# EXCLUDE FROM TESTING
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

@show backend, backend_str

const f_type = KernelAbstractions.supports_float64(backend) ? Float64 : Float32
