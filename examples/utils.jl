# EXCLUDE FROM TESTING
if Base.find_package("CUDA") !== nothing
    using CUDA
    using CUDA.CUDAKernels
    const has_cuda = true
else
    const has_cuda = false
end
if Base.find_package("AMDGPU") !== nothing
    using AMDGPU
    using ROCKernels
    const has_rocm = true
else
    const has_rocm = false
end

