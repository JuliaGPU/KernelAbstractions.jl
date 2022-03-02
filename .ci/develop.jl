import Pkg

Pkg.update()

root_directory = dirname(@__DIR__)

kernelabstractions = Pkg.PackageSpec(path = root_directory)

BACKEND = get(ENV, "KERNELABSTRACTIONS_TEST_BACKEND", "all")
CI = parse(Bool, get(ENV, "CI", "false"))

@info "Develop..." CI

Pkg.develop(kernelabstractions)
if !CI || BACKEND == "ROCM"
    rockernels = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "ROCKernels"))
    Pkg.develop(rockernels)
end

if !CI || BACKEND == "CUDA"
    cudakernels = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "CUDAKernels"))
    Pkg.develop(cudakernels)
end

if VERSION < v"1.8"
    kernelgradients = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "KernelGradients"))
    Pkg.develop(kernelgradients)
end
Pkg.build()
Pkg.precompile()
