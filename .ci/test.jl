import Pkg

pkgs = [
    "KernelAbstractions",
    "CUDAKernels",
    "ROCKernels",
]

Pkg.test(pkgs; coverage = true)
