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
using LLVM, LLVM.Interop
import LLVM: LLVM, MDNode, ConstantInt, metadata
using SPIRV_LLVM_Backend_jll, SPIRV_Tools_jll
using Adapt

## device overrides
import SPIRVIntrinsics
SPIRVIntrinsics.@import_all
SPIRVIntrinsics.@reexport_public
# local method table for device functions
Base.Experimental.@MethodTable(method_table)

import Core: LLVMPtr

# the device code comes first: the compiler's generated functions use its types, and a
# generator only sees the bindings that existed when it was defined
include("device/array.jl")
include("device/quirks.jl")
include("device/runtime.jl")
include("device/random.jl")

include("compiler/compilation.jl")
include("compiler/execution.jl")
include("compiler/reflection.jl")

function Adapt.adapt_storage(to::KernelAdaptor, xs::Array{T, N}) where {T, N}
    return CLDeviceArray{T, N, AS.CrossWorkgroup}(size(xs), reinterpret(LLVMPtr{T, AS.CrossWorkgroup}, pointer(xs)))
end

import KernelInterface
include("backend.jl")
import .POCLKernels: POCLBackend
export POCLBackend

import KernelAbstractions as KA

function __init__()
    initialization_world[] = Base.get_world_counter()
    return
end

# drop session-local state created by a precompilation workload
function reset_session_state!()
    empty!(_compiler_configs)
    empty!(_kernel_instances)
    return
end

end
