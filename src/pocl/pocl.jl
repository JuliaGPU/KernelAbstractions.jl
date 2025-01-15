module POCL

function platform end
function device end
function context end
function queue end

include("nanoOpenCL.jl")

import .nanoOpenCL as cl

function platform()
    return get!(task_local_storage(), :POCLPlatform) do
        for p in cl.platforms()
            if p.vendor == "The pocl project"
                return p
            end
        end
        error("POCL not available")
    end::cl.Platform
end

function device()
    return get!(task_local_storage(), :POCLDevice) do
        p = platform()
        return cl.default_device(p)
    end::cl.Device
end

# TODO: add a device context dict
function context()
    return get!(task_local_storage(), :POCLContext) do
        cl.Context(device())
    end::cl.Context
end

function queue()
    return get!(task_local_storage(), :POCLQueue) do
        cl.CmdQueue()
    end::cl.CmdQueue
end

using GPUCompiler
import LLVM
using Adapt

import SPIRVIntrinsics
SPIRVIntrinsics.@import_all
SPIRVIntrinsics.@reexport_public

include("compiler/compilation.jl")
include("compiler/execution.jl")
include("compiler/reflection.jl")

import Core: LLVMPtr

include("device/array.jl")
include("device/quirks.jl")
include("device/runtime.jl")

function Adapt.adapt_storage(to::KernelAdaptor, xs::Array{T, N}) where {T, N}
    return CLDeviceArray{T, N, AS.Global}(size(xs), reinterpret(LLVMPtr{T, AS.Global}, pointer(xs)))
end

include("backend.jl")
import .POCLKernels: POCLBackend
export POCLBackend

import KernelAbstractions as KA

Adapt.adapt_storage(::POCLBackend, a::Array) = a

end
