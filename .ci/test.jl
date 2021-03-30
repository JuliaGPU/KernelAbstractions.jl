import Pkg

pkgs = [
    "KernelAbstractions",
    "CUDAKernels",
]
if !(VERSION < v"1.6-")
    push!(pkgs, "ROCKernels")
end

Pkg.test(pkgs; coverage = true)
