module ROCKernels

import AMDGPU
import AMDGPU: rocfunction, ROCDevice, ROCQueue, ROCSignal, ROCKernelSignal, ROCArray, Mem
import StaticArrays
import StaticArrays: MArray
import Adapt
import KernelAbstractions

import UnsafeAtomicsLLVM

export ROCDevice

KernelAbstractions.isgpu(::ROCDevice) = true

KernelAbstractions.get_device(A::ROCArray) = AMDGPU.device(A)


const FREE_QUEUES = ROCQueue[]
const QUEUES = ROCQueue[]
const QUEUE_GC_THRESHOLD = Ref{Int}(16)

# This code is loaded after an `@init` step
if haskey(ENV, "KERNELABSTRACTIONS_QUEUES_GC_THRESHOLD")
    global QUEUE_GC_THRESHOLD[] = parse(Int, ENV["KERNELABSTRACTIONS_QUEUES_GC_THRESHOLD"])
end

## Stream GC
# Simplistic stream gc design in which when we have a total number
# of streams bigger than a threshold, we start scanning the streams
# and add them back to the freelist if all work on them has completed.
# Alternative designs:
# - Enqueue a host function on the stream that adds the stream back to the freelist
# - Attach a finalizer to events that adds the stream back to the freelist
# Possible improvements
# - Add a background task that occasionally scans all streams
# - Add a hysterisis by checking a "since last scanned" timestamp
# - Add locking
const QUEUE_GC_LOCK = Threads.ReentrantLock()
function next_queue()
    lock(QUEUE_GC_LOCK) do
        if !isempty(FREE_QUEUES)
            return pop!(FREE_QUEUES)
        end

        if length(QUEUES) > QUEUE_GC_THRESHOLD[]
            for queue in QUEUES
                #if AMDGPU.queued_packets(queue) == 0
                    push!(FREE_QUEUES, queue)
                #end
            end
        end

        if !isempty(FREE_QUEUES)
            return pop!(FREE_QUEUES)
        end

        # FIXME: Which device?
        queue = ROCQueue()
        push!(QUEUES, queue)
        return queue
    end
end

import KernelAbstractions: Event, CPUEvent, NoneEvent, MultiEvent, CPU, GPU, isdone, failed, construct

struct ROCEvent{T<:Union{ROCKernelSignal,ROCSignal,AMDGPU.HSA.Signal}} <: Event
    event::T
end

cpuwait(ev::ROCEvent{ROCSignal}) = wait(ev.event)
cpuwait(ev::ROCEvent{ROCKernelSignal}) = wait(ev.event)

gpuwait(ev::ROCEvent{AMDGPU.HSA.Signal}) = AMDGPU.Device.hostcall_device_signal_wait(ev.event.handle, 0)
#gpuwait(ev::ROCEvent{ROCSignal}) = wait(ev.event)
#gpuwait(ev::ROCEvent{ROCKernelSignal}) = wait(ev.event)

Adapt.adapt_storage(::AMDGPU.Runtime.Adaptor, ev::ROCEvent{ROCKernelSignal}) =
    ROCEvent(ev.event.signal.signal[])

failed(ev::ROCEvent{ROCKernelSignal}) =
    isdone(ev) && ev.event.exception !== nothing
isdone(ev::ROCEvent{ROCKernelSignal}) =
    ev.event.done.set # TODO: Don't access .set

failed(ev::ROCEvent{ROCSignal}) =
    false # FIXME: We can't know this from just the signal value
isdone(ev::ROCEvent{ROCSignal}) =
    AMDGPU.HSA.hsa_signal_load_scacquire(ev.event.signal[]) < 1

function Event(::ROCDevice)
    queue = AMDGPU.default_queue()
    # Returns a ROCSignalSet containing signals
    event = AMDGPU.barrier_and!(queue, AMDGPU.active_kernels(queue))
    # Build ROCEvents and put them in a MultiEvent
    MultiEvent(Tuple(ROCEvent(s) for s in event.signals))
end

import Base: wait

wait(ev::ROCEvent, progress=nothing) = wait(CPU(), ev, progress)

function wait(::CPU, ev::ROCEvent, progress=nothing)
    if progress === nothing
        cpuwait(ev)
    else
        # FIXME: One-shot wait
        while !cpuwait(ev)
            progress()
            # do we need to `yield` here?
        end
    end
end

wait(::ROCDevice, ev::ROCEvent, progress=nothing, queue=nothing) = cpuwait(ev) #gpuwait(ev)
wait(::ROCDevice, ev::NoneEvent, progress=nothing, queue=nothing) = nothing

function wait(::ROCDevice, ev::MultiEvent, progress=nothing, queue=AMDGPU.default_queue())
    dependencies = collect(ev.events)
    rocdeps = map(d->d.event isa ROCKernelSignal ? d.event.signal.signal[] : d.event, filter(d->d isa ROCEvent, dependencies))
    wait.(rocdeps)
    #isempty(rocdeps) || wait(AMDGPU.barrier_and!(queue, rocdeps))
    for event in filter(d->!(d isa ROCEvent), dependencies)
        wait(queue.device, event, progress)
    end
end

# TODO: Create a hostcall that waits on a specified Task
wait(::ROCDevice, ev::CPUEvent, progress=nothing, queue=nothing) = error("GPU->CPU wait not implemented")

function KernelAbstractions.async_copy!(dev::ROCDevice, A::ROCArray{TA}, B::ROCArray{TB}; dependencies=nothing, progress=nothing) where {TA, TB}
    # TODO: Register A and B
    wait(dev, MultiEvent(dependencies), progress, AMDGPU.default_queue(dev))
    GC.@preserve A B begin
        destptr = pointer(A)
        srcptr  = pointer(B)
        N       = length(A)
        signal  = ROCSignal()
        #if TA === TB
        #    AMDGPU.Mem.unsafe_copy3d!(destptr, srcptr, N; signal, async=true)
        #else
            # TODO: We should still be able to unsafe_copy3d with some checks
            AMDGPU.HSA.memory_copy(destptr, srcptr, N) |> AMDGPU.Runtime.check
            notify(signal)
        #end
        return ROCEvent(signal)
    end

end

import KernelAbstractions: Kernel, StaticSize, DynamicSize, partition, blocks, workitems, launch_config

###
# Kernel launch
###
function launch_config(kernel::Kernel{ROCDevice}, ndrange, workgroupsize)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize, )
    end

    # partition checked that the ndrange's agreed
    if KernelAbstractions.ndrange(kernel) <: StaticSize
        ndrange = nothing
    end

    iterspace, dynamic = if KernelAbstractions.workgroupsize(kernel) <: DynamicSize && workgroupsize === nothing
        workgroupsize = ntuple(i->i == 1 ? min(prod(ndrange), AMDGPU.Device._max_group_size) : 1, length(ndrange))
        partition(kernel, ndrange, workgroupsize)
    else
        partition(kernel, ndrange, workgroupsize)
    end

    return ndrange, workgroupsize, iterspace, dynamic
end

function threads_to_workgroupsize(threads, ndrange)
    total = 1
    return map(ndrange) do n
        x = min(div(threads, total), n)
        total *= x
        return x
    end
end

function construct(dev::ROCDevice, sz::S, range::NDRange, gpu_name::GPUName) where {S<:KernelAbstractions._Size, NDRange<:KernelAbstractions._Size, GPUName}
    return Kernel{ROCDevice, S, NDRange, GPUName}(gpu_name)
end

function (obj::Kernel{ROCDevice})(args...; ndrange=nothing, dependencies=nothing, workgroupsize=nothing, progress=nothing)
    ndrange, workgroupsize, iterspace, dynamic = launch_config(obj, ndrange, workgroupsize)
    # this might not be the final context, since we may tune the workgroupsize
    ctx = mkcontext(obj, ndrange, iterspace)
    #= TODO: Autotuning
    kernel = AMDGPU.@roc launch=false name=String(nameof(obj.f)) Cassette.overdub(ctx, obj.f, args...)

    # figure out the optimal workgroupsize automatically
    if KernelAbstractions.workgroupsize(obj) <: DynamicSize && workgroupsize === nothing
        config = AMDGPU.launch_configuration(kernel.fun; max_threads=prod(ndrange))
        workgroupsize = threads_to_workgroupsize(config.threads, ndrange)
        iterspace, dynamic = partition(obj, ndrange, workgroupsize)
        ctx = mkcontext(obj, ndrange, iterspace)
    end

    # If the kernel is statically sized we can tell the compiler about that
    if KernelAbstractions.workgroupsize(obj) <: StaticSize
        maxthreads = prod(KernelAbstractions.get(KernelAbstractions.workgroupsize(obj)))
    else
        maxthreads = nothing
    end
    =#

    nblocks = length(blocks(iterspace))
    nthreads = length(workitems(iterspace))

    if nblocks == 0
        return MultiEvent(dependencies)
    end

    queue = next_queue()
    device = queue.device
    wait(device, MultiEvent(dependencies), progress, queue)

    # Launch kernel
    event = AMDGPU.@roc(groupsize=nthreads,
                        gridsize=nblocks*nthreads,
                        queue=queue,
                        name=String(nameof(obj.f)),
                        # TODO: maxthreads=maxthreads,
                        obj.f(ctx, args...))

    return ROCEvent(event)
end

import AMDGPU.Device: @device_override

import KernelAbstractions: CompilerMetadata, DynamicCheck, LinearIndices
import KernelAbstractions: __index_Local_Linear, __index_Group_Linear, __index_Global_Linear, __index_Local_Cartesian, __index_Group_Cartesian, __index_Global_Cartesian, __validindex, __print
import KernelAbstractions: mkcontext, expand, __iterspace, __ndrange, __dynamic_checkbounds

function mkcontext(kernel::Kernel{ROCDevice}, _ndrange, iterspace)
    metadata = CompilerMetadata{KernelAbstractions.ndrange(kernel), DynamicCheck}(_ndrange, iterspace)
end
function mkcontext(kernel::Kernel{ROCDevice}, I, _ndrange, iterspace, ::Dynamic) where Dynamic
    metadata = CompilerMetadata{KernelAbstractions.ndrange(kernel), Dynamic}(I, _ndrange, iterspace)
end

@device_override @inline function __index_Local_Linear(ctx)
    return AMDGPU.Device.threadIdx().x
end

@device_override @inline function __index_Group_Linear(ctx)
    return AMDGPU.Device.blockIdx().x
end

@device_override @inline function __index_Global_Linear(ctx)
    I =  @inbounds expand(__iterspace(ctx), AMDGPU.Device.blockIdx().x, AMDGPU.Device.threadIdx().x)
    # TODO: This is unfortunate, can we get the linear index cheaper
    @inbounds LinearIndices(__ndrange(ctx))[I]
end

@device_override @inline function __index_Local_Cartesian(ctx)
    @inbounds workitems(__iterspace(ctx))[AMDGPU.Device.threadIdx().x]
end

@device_override @inline function __index_Group_Cartesian(ctx)
    @inbounds blocks(__iterspace(ctx))[AMDGPU.Device.blockIdx().x]
end

@device_override @inline function __index_Global_Cartesian(ctx)
    return @inbounds expand(__iterspace(ctx), AMDGPU.Device.blockIdx().x, AMDGPU.Device.threadIdx().x)
end

@device_override @inline function __validindex(ctx)
    if __dynamic_checkbounds(ctx)
        I = @inbounds expand(__iterspace(ctx), AMDGPU.Device.blockIdx().x, AMDGPU.Device.threadIdx().x)
        return I in __ndrange(ctx)
    else
        return true
    end
end

import KernelAbstractions: groupsize, __groupsize, __workitems_iterspace, add_float_contract, sub_float_contract, mul_float_contract
import KernelAbstractions: ConstAdaptor, SharedMemory, Scratchpad, __synchronize, __size

###
# GPU implementation of shared memory
# - shared memory for each workgroup
###
@device_override @inline function SharedMemory(::Type{T}, ::Val{Dims}, ::Val{Id}, ::Val{Zeroinit}) where {T, Dims, Id, Zeroinit}
    ptr = AMDGPU.Device.alloc_special(Val(Id), T, Val(AMDGPU.AS.Local), Val(prod(Dims)), Val(Zeroinit))
    AMDGPU.ROCDeviceArray(Dims, ptr)
end

###
# GPU implementation of scratch memory
# - private memory for each workitem
###

@device_override @inline function Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    MArray{__size(Dims), T}(undef)
end

@device_override @inline function __synchronize()
    AMDGPU.Device.sync_workgroup()
end

@device_override @inline function __print(args...)
    for arg in args
        AMDGPU.Device.@rocprintf("%s", arg)
    end
end

###
# GPU implementation of constant memory
# - access to constant (unchanging) memory
###

Adapt.adapt_storage(to::ConstAdaptor, a::AMDGPU.ROCDeviceArray{T,N,A}) where {T,N,A} =
    AMDGPU.ROCDeviceArray(a.shape, reinterpret(Core.LLVMPtr{T,AMDGPU.Device.AS.Constant}, a.ptr))

# Argument conversion

KernelAbstractions.argconvert(::Kernel{ROCDevice}, arg) = AMDGPU.rocconvert(arg)

end
