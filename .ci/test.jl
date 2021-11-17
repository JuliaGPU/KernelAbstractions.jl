import Pkg

BACKEND = get(ENV, "KERNELABSTRACTIONS_TEST_BACKEND", "all")
BUILDKITE = parse(Bool, get(ENV, "BUILDKITE", "false"))

pkgs = [
    "KernelAbstractions",
]
if !(VERSION < v"1.6-")
    if !BUILDKITE  || BACKEND == "ROCM"
        push!(pkgs, "ROCKernels")
    end
    if !BUILDKITE  || BACKEND == "CUDA"
        push!(pkgs, "CUDAKernels")
    end
    push!(pkgs, "KernelGradients")
end

Pkg.test(pkgs; coverage = true)
