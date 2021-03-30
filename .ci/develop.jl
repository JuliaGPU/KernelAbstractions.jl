import Pkg

Pkg.update()

root_directory = dirname(@__DIR__)

kernelabstractions = Pkg.PackageSpec(path = root_directory)
cudakernels = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "CUDAKernels"))

Pkg.develop(kernelabstractions)
Pkg.develop(cudakernels)
if !(VERSION < v"1.6-")
    rockernels = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "ROCKernels"))
    Pkg.develop(rockernels)
end
Pkg.build()
Pkg.precompile()
