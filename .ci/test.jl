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
push!(pkgs, "KernelGradients")

Pkg.test(pkgs; coverage = true)
