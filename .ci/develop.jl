import Pkg

Pkg.update()

root_directory = dirname(@__DIR__)

kernelabstractions = Pkg.PackageSpec(path = root_directory)

BACKEND = get(ENV, "KERNELABSTRACTIONS_TEST_BACKEND", "all")
BUILDKITE = parse(Bool, get(ENV, "BUILDKITE", "false"))

@info "Develop..." BUILDKITE

Pkg.develop(kernelabstractions)
if !(VERSION < v"1.6-")
    if !BUILDKITE  || BACKEND == "ROCM"
        rockernels = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "ROCKernels"))
        Pkg.develop(rockernels)
    end

    if !BUILDKITE  || BACKEND == "CUDA"
        cudakernels = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "CUDAKernels"))
        Pkg.develop(cudakernels)
    end

    kernelgradients = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "KernelGradients"))
    Pkg.develop(kernelgradients)
end
Pkg.build()
Pkg.precompile()
