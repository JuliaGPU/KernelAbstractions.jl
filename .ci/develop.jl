import Pkg

root_directory = dirname(@__DIR__)

kernelabstractions = Pkg.PackageSpec(path = root_directory)
cudakernels = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "CUDAKernels"))
rockernels = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "ROCKernels"))

Pkg.develop(kernelabstractions)
Pkg.develop(cudakernels)
Pkg.develop(rockernels)
Pkg.build(rockernels)

Pkg.precompile()
