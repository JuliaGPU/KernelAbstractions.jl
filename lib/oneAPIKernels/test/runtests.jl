using KernelAbstractions
using oneAPI
using oneAPIKernels
using Test
using SparseArrays

include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))

CI = parse(Bool, get(ENV, "CI", "false"))
if CI
    default = "CPU"
else
    default = "oneAPI"
end

backend = get(ENV, "KERNELABSTRACTIONS_TEST_BACKEND", default)

if backend != "oneAPI"
    @info "oneAPI backend not selected"
    exit()
end

oneAPI.versioninfo()
oneAPI.allowscalar(false)
Testsuite.testsuite(oneAPIDevice, backend, oneAPI, oneArray, oneAPI.oneDeviceArray)
