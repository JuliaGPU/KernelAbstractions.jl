import Pkg

Pkg.update()

root_directory = dirname(@__DIR__)

kernelabstractions = Pkg.PackageSpec(path = root_directory)

BACKEND = get(ENV, "KERNELABSTRACTIONS_TEST_BACKEND", "all")
CI = parse(Bool, get(ENV, "CI", "false"))

@info "Develop..." CI

pkgs = [kernelabstractions]

if !CI || BACKEND == "ROCM"
    rockernels = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "ROCKernels"))
    push!(pkgs, rockernels)
end

if !CI || BACKEND == "CUDA"
    cudakernels = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "CUDAKernels"))
    push!(pkgs, cudakernels)
end

if !CI || BACKEND == "oneAPI"
    oneapikernels = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "oneAPIKernels"))
    push!(pkgs, oneapikernels)
end

if !CI || BACKEND == "Metal"
    metalkernels = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "MetalKernels"))
    push!(pkgs, metalkernels)
    Pkg.add(Pkg.PackageSpec(name="Metal", rev="main")) # TODO remove when tagged
end

if VERSION < v"1.8"
    kernelgradients = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "KernelGradients"))
    push!(pkgs, kernelgradients)
end
Pkg.develop(pkgs)
Pkg.build()
Pkg.precompile()
