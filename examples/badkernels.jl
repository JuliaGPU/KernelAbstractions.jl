using KernelAbstractions
using CUDAapi
using Test

@kernel function return_obj()
    return 5.0
end

function check()
    x = return_obj(CUDA, 256)()
end

check()
