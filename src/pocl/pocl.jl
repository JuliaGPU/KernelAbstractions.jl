module POCL

function platform end
function device end
function context end
function queue end

include("nanoOpenCL.jl")

import .nanoOpenCL as cl

## session state

# what every task shares: one context, so that kernels are only linked once per process
struct Session
    platform::cl.Platform
    device::cl.Device
    context::cl.Context
    # querying the device allocates, so cache what launches need
    limits::@NamedTuple{
        max_work_group_size::Int, max_work_group_dims::NTuple{3, Int}, sub_group_size::Int,
        compute_units::Int, preferred_work_group_size_multiple::Int,
        fp16::Bool, fp64::Bool,
    }
end

# the sub-group width kernels are compiled for, or `nothing` if the device has no sub-groups.
# POCL can technically support any sub-group size; prefer the common GPU ones, and only
# widths of up to 64, so that `KI.sub_group_ballot` (a `UInt64` mask) covers every lane.
function default_sub_group_size(device::cl.Device)
    sg_sizes = filter(<=(64), device.sub_group_sizes::Tuple{Vararg{Int}})
    common = filter(in(sg_sizes), Int[32, 64, 16, sg_sizes...])
    return isempty(common) ? nothing : first(common)
end

function Session()
    cl.pocl_standalone_jll.is_available() ||
        error("The CPU back-end needs PoCL, which is not available for $(Sys.MACHINE)")
    idx = findfirst(p -> p.vendor == "The pocl project", cl.platforms())
    idx === nothing && error("POCL not available")
    platform = cl.platforms()[idx]
    device = cl.default_device(platform)
    # PoCL's thread pool can only be sized with environment variables, so use a sub-device
    # to run kernels on fewer threads. its other threads stay asleep.
    threads = cl.cpu_threads()
    if threads !== nothing && threads < device.max_compute_units && device.max_sub_devices > 0
        device = cl.sub_device(device, threads)
    end
    context = cl.Context(device)

    sizes = device.max_work_item_size
    extensions = device.extensions
    sub_group_size = default_sub_group_size(device)
    limits = (;
        max_work_group_size = Int(device.max_work_group_size),
        max_work_group_dims = ntuple(d -> d <= length(sizes) ? sizes[d] : 1, 3),
        # 0 if the device has no sub-groups
        sub_group_size = something(sub_group_size, 0),
        # the threads PoCL runs work-groups on
        compute_units = Int(device.max_compute_units),
        # what kernels report, unless PoCL chose another for them
        preferred_work_group_size_multiple = Int(device.preferred_work_group_size_multiple),
        # collectives check these on every launch
        fp16 = "cl_khr_fp16" in extensions,
        fp64 = "cl_khr_fp64" in extensions,
    )

    return Session(platform, device, context, limits)
end

# created on first use, so that loading the package doesn't initialize PoCL
mutable struct SessionCache
    Base.@atomic session::Union{Nothing, Session}
    const lock::ReentrantLock
end
const session_cache = SessionCache(nothing, ReentrantLock())

@inline function session()
    s = Base.@atomic :acquire session_cache.session
    s === nothing || return s
    return init_session()
end
@noinline function init_session()
    return @lock session_cache.lock begin
        s = Base.@atomic :acquire session_cache.session
        if s === nothing
            s = Session()
            Base.@atomic :release session_cache.session = s
        end
        s
    end::Session
end

platform() = session().platform
device() = session().device
context() = session().context
device_limits() = session().limits

# queues are per task, like streams on GPU back-ends, so that a task waiting for its
# kernel doesn't hold up kernels from other tasks
function queue()
    s = session()
    tls = task_local_storage()
    entry = get(tls, :POCLQueue, nothing)::Union{Nothing, Tuple{Session, cl.CmdQueue}}
    if entry !== nothing && entry[1] === s
        return entry[2]
    end
    q = cl.CmdQueue(s.context, s.device)
    tls[:POCLQueue] = (s, q)
    return q
end

using GPUCompiler
using LLVM, LLVM.Build, LLVM.Interop
# not all of LLVM.IR, whose `context` would clash with ours
using LLVM.IR: BasicBlock, ConstantArray, ConstantInt, DILocation, GlobalVariable, LLVMType,
    MDNode, MD_invariant_load, MD_tbaa, MemoryEffects, StringAttribute, Value
using SPIRV_LLVM_Backend_jll, SPIRV_Tools_jll
using Adapt

## device overrides
import SPIRVIntrinsics
SPIRVIntrinsics.@import_all
SPIRVIntrinsics.@reexport_public
# local method table for device functions
Base.Experimental.@MethodTable(method_table)

import Core: LLVMPtr
import UnsafeAtomics

# the device code comes first: the compiler's generated functions use its types, and a
# generator only sees the bindings that existed when it was defined
include("device/array.jl")
include("device/quirks.jl")
include("device/runtime.jl")
include("device/random.jl")
include("device/shuffle.jl")

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
    # there shouldn't be any session from precompilation, see `reset_session_state!`
    Base.@atomic session_cache.session = nothing
    return
end

# drop session-local state created by a precompilation workload, so that no handles to
# OpenCL objects get serialized. kernels launched before the reset can't be used anymore,
# so this must not race with other tasks using the back-end.
function reset_session_state!()
    @lock clfunction_lock empty!(_compiler_configs)
    @lock session_cache.lock begin
        Base.@atomic session_cache.session = nothing
    end
    delete!(task_local_storage(), :POCLQueue)
    delete!(task_local_storage(), :POCLExceptionInfo)
    return
end

end
