import Pkg

pkgs = [
    "KernelAbstractions",
    "CUDAKernels",
]

Pkg.test(pkgs; coverage = true)
