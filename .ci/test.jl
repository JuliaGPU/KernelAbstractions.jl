import Pkg

pkgs = [
    "KernelAbstractions",
]
if !(VERSION < v"1.6-")
    push!(pkgs, "ROCKernels")
    push!(pkgs, "CUDAKernels")
    push!(pkgs, "KernelGradients")
end

Pkg.test(pkgs; coverage = true)
