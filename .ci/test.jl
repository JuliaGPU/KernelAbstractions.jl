import Pkg

BACKEND = get(ENV, "KERNELABSTRACTIONS_TEST_BACKEND", "all")
CI = parse(Bool, get(ENV, "CI", "false"))

pkgs = [
    "KernelAbstractions",
]
if !CI || BACKEND == "ROCM"
    push!(pkgs, "ROCKernels")
end
if !CI || BACKEND == "CUDA"
    push!(pkgs, "CUDAKernels")
end
if !CI || BACKEND == "oneAPI"
    push!(pkgs, "oneAPIKernels")
end
if !CI || haskey(ENV, "TEST_KERNELGRADIENTS")
    if VERSION < v"1.8"
        push!(pkgs, "KernelGradients")
    end
end

Pkg.test(pkgs; coverage = true)
