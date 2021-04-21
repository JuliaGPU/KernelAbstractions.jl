import Pkg

Pkg.update()

root_directory = dirname(@__DIR__)

kernelabstractions = Pkg.PackageSpec(path = root_directory)

Pkg.develop(kernelabstractions)
if !(VERSION < v"1.6-")
    rockernels = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "ROCKernels"))
    Pkg.develop(rockernels)

    cudakernels = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "CUDAKernels"))
    Pkg.develop(cudakernels)
end
Pkg.build()
Pkg.precompile()
