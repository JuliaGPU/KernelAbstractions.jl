import Pkg

root_directory = dirname(@__DIR__)

kernelabstractions = Pkg.PackageSpec(path = root_directory)
cudakernels = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "CUDAKernels"))

Pkg.develop(kernelabstractions)
Pkg.develop(cudakernels)

Pkg.precompile()
