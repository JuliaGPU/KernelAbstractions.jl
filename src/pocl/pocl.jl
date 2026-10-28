module POCL

function platform end
function device end
function context end
function queue end

include("nanoOpenCL.jl")

import .nanoOpenCL as cl

## session state

# shared context: link kernels once per process.
# mutable so `session()` returns a reference, avoiding repeated copies per launch.
mutable struct Session
    const platform::cl.Platform
    const device::cl.Device
    const context::cl.Context
    # querying the device allocates, so cache what launches need
    const limits::@NamedTuple{
        max_work_group_size::Int, max_work_group_dims::NTuple{3, Int}, sub_group_size::Int,
        compute_units::Int, preferred_work_group_size_multiple::Int,
        fp16::Bool, fp64::Bool,
    }
    # duration threshold in cycle counter ticks for long launches (see `host_dispatch`)
    const host_kernel_long_ticks::UInt64
    # serial-duration threshold in ticks for small launches (see `host_schedule`)
    const host_kernel_serial_ticks::UInt64
    # below this duration (ticks), sample timings on the launching task; see `run_short`
    const host_kernel_short_ticks::UInt64
end

# a cheap monotonic counter: the TSC on x86, the virtual timer on AArch64 (where
# `llvm.readcyclecounter` reads PMCCNTR_EL0, which user code usually may not access)
@static if Sys.ARCH === :x86_64 || Sys.ARCH === :i686
    @inline cycle_counter() = ccall("llvm.readcyclecounter", llvmcall, UInt64, ())
elseif Sys.ARCH === :aarch64
    @inline cycle_counter() = Base.llvmcall(
        """
        %t = call i64 asm sideeffect "mrs \$0, cntvct_el0", "=r"()
        ret i64 %t
        """, UInt64, Tuple{}
    )
else
    @inline cycle_counter() = time_ns()
end

function ticks_per_ns()
    t0, c0 = time_ns(), cycle_counter()
    while time_ns() - t0 < 1_000_000
    end
    t1, c1 = time_ns(), cycle_counter()
    return (c1 - c0) / (t1 - t0)
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
    # with the requested number of compute units. its other threads stay asleep.
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
        # cap host-kernel threads at Julia's default pool size
        compute_units = min(Int(device.max_compute_units), Threads.nthreads(:default)),
        # what kernels report, unless PoCL chose another for them
        preferred_work_group_size_multiple = Int(device.preferred_work_group_size_multiple),
        # collectives check these on every launch
        fp16 = "cl_khr_fp16" in extensions,
        fp64 = "cl_khr_fp64" in extensions,
    )

    "cl_pocl_host_kernel" in extensions ||
        error("The CPU back-end needs a PoCL build with the cl_pocl_host_kernel extension")

    rate = ticks_per_ns()
    # 50 µs, which is long compared to the cost of letting other tasks run
    host_kernel_long_ticks = round(UInt64, 50_000 * rate)
    # 4 µs, several times what starting and joining a team costs, see `host_schedule`
    host_kernel_serial_ticks = round(UInt64, 4_000 * rate)
    # 1 µs, compared with tens of nanoseconds to time a launch
    host_kernel_short_ticks = round(UInt64, 1_000 * rate)

    return Session(
        platform, device, context, limits, host_kernel_long_ticks,
        host_kernel_serial_ticks, host_kernel_short_ticks
    )
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
    # nor scratch areas, whose pointers don't survive serialization
    Base.@atomic host_scratch.slots = HostScratch[]
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
