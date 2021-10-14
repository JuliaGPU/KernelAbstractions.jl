using KernelAbstractions
using KernelGradients
using Enzyme
using CUDA
using CUDAKernels
using Test
using SparseArrays

include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))
include(joinpath(dirname(pathof(KernelGradients)), "..", "test", "testsuite.jl"))

@testset "get_device" begin
    @test @inferred(KernelAbstractions.get_device(CuArray(rand(Float32, 3,3)))) == CUDADevice()
    @test @inferred(KernelAbstractions.get_device(CuArray(sparse(rand(Float32, 3,3))))) == CUDADevice()
end

CI = parse(Bool, get(ENV, "CI", "false"))
if CI
    default = "CPU"
else
    default = "CUDA"
end

backend = get(ENV, "KERNELABSTRACTIONS_TEST_BACKEND", default)

if backend != "CUDA"
    @info "CUDA backend not selected"
    exit()
end

# if CUDA.functional(true)
if CUDA.functional()
    CUDA.versioninfo()
    CUDA.allowscalar(false)
    Testsuite.testsuite(CUDADevice, backend, CUDA, CuArray, CUDA.CuDeviceArray)
    # GradientsTestsuite.testsuite(CUDADevice, backend, CUDA, CuArray, CUDA.CuDeviceArray)
elseif !CI
    error("No CUDA GPUs available!")
end
