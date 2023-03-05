module MetalKernels

import Metal
import Metal: MTL
import StaticArrays
import KernelAbstractions

export MetalDevice

KernelAbstractions.get_device(::Metal.MtlArray) = MetalDevice()


const FREE_QUEUES = Metal.MtlCommandQueue[]
const QUEUES = Metal.MtlCommandQueue[]
const QUEUE_GC_THRESHOLD = Ref{Int}(16)

# This code is loaded after an `@init` step
if haskey(ENV, "KERNELABSTRACTIONS_QUEUES_GC_THRESHOLD")
    global QUEUE_GC_THRESHOLD[] = parse(Int, ENV["KERNELABSTRACTIONS_QUEUES_GC_THRESHOLD"])
end

## Queue GC
# Simplistic queue gc design in which when we have a total number
# of queues bigger than a threshold, we start scanning the queues
# and add them back to the freelist if all work on them has completed.
# Alternative designs:
# - Enqueue a host function on the stream that adds the stream back to the freelist
# - Attach a finalizer to events that adds the stream back to the freelist
# Possible improvements
# - Add a background task that occasionally scans all queues
# - Add a hysterisis by checking a "since last scanned" timestamp
# - Add locking
const QUEUE_GC_LOCK = Threads.ReentrantLock()
function next_queue()
    dev = Metal.current_device()
    lock(QUEUE_GC_LOCK) do
        if !isempty(FREE_QUEUES)
            return pop!(FREE_QUEUES)
        end

        if length(QUEUES) > QUEUE_GC_THRESHOLD[]
            for queue in QUEUES
                # XXX: can we check if the queue is idle?
                 push!(FREE_QUEUES, queue)
            end
        end

        if !isempty(FREE_QUEUES)
            return pop!(FREE_QUEUES)
        end

        queue = Metal.MtlCommandQueue(dev)
        push!(QUEUES, queue)
        return queue
    end
end

function default_queue()
    dev = Metal.current_device()
    Metal.global_queue(dev)
end

import KernelAbstractions: Event, CPUEvent, NoneEvent, MultiEvent, CPU, GPU, isdone, failed

struct MetalDevice <: GPU end

const METAL_EVENT_SIGNAL_VALUE = UInt64(2)

struct MetalEvent <: Event
    event::Metal.MtlSharedEvent
end

failed(::MetalEvent) = false
isdone(ev::MetalEvent) = ev.event.signaledValue == METAL_EVENT_SIGNAL_VALUE

function Event(::MetalDevice)
    device = Metal.current_device()
    queue = Metal.global_queue(device)
    cmdbuf = Metal.MtlCommandBuffer(queue)
    event = Metal.MtlSharedEvent(device)

    # The cmdbuf executes in-order on the global queue
    # so this one just triggers our signal
    MTL.encode_signal!(cmdbuf, event, METAL_EVENT_SIGNAL_VALUE)
    Metal.commit!(cmdbuf)

    MetalEvent(event)
end

import Base: wait

wait(ev::MetalEvent, progress=nothing) = wait(CPU(), ev, progress)


function wait(::CPU, ev::MetalEvent, progress=nothing)
    buf = Metal.MtlCommandBuffer(next_queue())
    Metal.encode_wait!(buf, ev.event, METAL_EVENT_SIGNAL_VALUE)
    Metal.commit!(buf)
    Metal.wait_completed(buf)
end


function wait(::MetalDevice, ev::MetalEvent, progress=nothing; queue::Metal.MtlCommandQueue=Metal.global_queue(Metal.current_device()))
    buf = Metal.MtlCommandBuffer(queue)
    Metal.enqueue!(buf)
    Metal.encode_wait!(buf, ev.event, METAL_EVENT_SIGNAL_VALUE)
    Metal.commit!(buf)
end

wait(::MetalDevice, ev::NoneEvent, progress=nothing; queue=nothing) = nothing

function wait(::MetalDevice, ev::MultiEvent, progress=nothing; queue::Metal.MtlCommandQueue=Metal.global_queue(Metal.current_device()))
    dependencies = collect(ev.events)
    metaldeps  =  filter(d->d isa MetalEvent,  dependencies)
    otherdeps = filter(d->!(d isa MetalEvent), dependencies)

    buf = Metal.MtlCommandBuffer(queue)
    Metal.enqueue!(buf)
    
    for ev in metaldeps
        Metal.encode_wait!(buf, ev.event, METAL_EVENT_SIGNAL_VALUE)
    end

    Metal.commit!(buf)

    for ev in otherdeps
        wait(MetalDevice(), ev, progress; queue)
    end
end

wait(::MetalDevice, ev::CPUEvent, progress=nothing; queue=nothing) = error("GPU->CPU wait not implemented")


function KernelAbstractions.async_copy!(::MetalDevice, A, B; dependencies=nothing, progress=nothing)
    queue = next_queue()
    dev = Metal.current_device()
    wait(MetalDevice(), MultiEvent(dependencies), progress; queue)

    event = Metal.MtlSharedEvent(dev)  ### FIXME: Event vs SharedEvent
    cmdbuf = Metal.MtlCommandBuffer(queue)
    
    dst = pointer(A)
    src = pointer(B)
    N = length(A)

    unsafe_copyto!(dev, dst, src, N, queue=queue, async=true)

    MTL.encode_signal!(cmdbuf, event, METAL_EVENT_SIGNAL_VALUE)
    Metal.commit!(cmdbuf)

    return MetalEvent(event)
end

import KernelAbstractions: Kernel, StaticSize, DynamicSize, partition, blocks, workitems, launch_config

###
# Kernel launch
###
function launch_config(kernel::Kernel{MetalDevice}, ndrange, workgroupsize)
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

    iterspace, dynamic = if KernelAbstractions.workgroupsize(kernel) <: DynamicSize &&
        workgroupsize === nothing
        # use ndrange as preliminary workgroupsize for autotuning
        partition(kernel, ndrange, ndrange)
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

function (obj::Kernel{MetalDevice})(args...; ndrange=nothing, dependencies=Event(MetalDevice()), workgroupsize=nothing, progress=nothing)
    ndrange, workgroupsize, iterspace, dynamic = launch_config(obj, ndrange, workgroupsize)
    # this might not be the final context, since we may tune the workgroupsize
    ctx = mkcontext(obj, ndrange, iterspace)
    kernel = Metal.@metal launch=false obj.f(ctx, args...)

    # TODO: Autotuning

    nblocks = length(blocks(iterspace))
    threads = length(workitems(iterspace))

    if nblocks == 0
        return MultiEvent(dependencies)
    end

    queue = next_queue()
    wait(MetalDevice(), MultiEvent(dependencies), progress; queue)

    event = Metal.MtlSharedEvent(Metal.current_device())  ### FIXME: Event vs SharedEvent

    # Launch kernel
    kernel(ctx, args...; threads=threads, grid=nblocks, queue=queue)

    buf = Metal.MtlCommandBuffer(queue)
    Metal.enqueue!(buf)
    Metal.encode_signal!(buf, event, METAL_EVENT_SIGNAL_VALUE)
    Metal.commit!(buf)

    return MetalEvent(event)
end

import Metal: @device_override

import KernelAbstractions: CompilerMetadata, DynamicCheck, LinearIndices
import KernelAbstractions: __index_Local_Linear, __index_Group_Linear, __index_Global_Linear, __index_Local_Cartesian, __index_Group_Cartesian, __index_Global_Cartesian, __validindex, __print
import KernelAbstractions: mkcontext, expand, __iterspace, __ndrange, __dynamic_checkbounds

function mkcontext(kernel::Kernel{MetalDevice}, _ndrange, iterspace)
    metadata = CompilerMetadata{KernelAbstractions.ndrange(kernel), DynamicCheck}(_ndrange, iterspace)
end
function mkcontext(kernel::Kernel{MetalDevice}, I, _ndrange, iterspace, ::Dynamic) where Dynamic
    metadata = CompilerMetadata{KernelAbstractions.ndrange(kernel), Dynamic}(I, _ndrange, iterspace)
end

@device_override @inline function __index_Local_Linear(ctx)
    return Metal.thread_position_in_threadgroup_1d()
end

@device_override @inline function __index_Group_Linear(ctx)
    return Metal.threadgroup_position_in_grid_1d()
end

@device_override @inline function __index_Global_Linear(ctx)
    return Metal.thread_position_in_grid_1d()
end

@device_override @inline function __index_Local_Cartesian(ctx)
    @inbounds workitems(__iterspace(ctx))[Metal.thread_position_in_threadgroup_1d()]
end

@device_override @inline function __index_Group_Cartesian(ctx)
    @inbounds blocks(__iterspace(ctx))[Metal.threadgroup_position_in_grid_1d()]
end

@device_override @inline function __index_Global_Cartesian(ctx)
    return @inbounds expand(__iterspace(ctx), Metal.threadgroup_position_in_grid_1d(), Metal.thread_position_in_threadgroup_1d())
end

@device_override @inline function __validindex(ctx)
    if __dynamic_checkbounds(ctx)
        I = @inbounds expand(__iterspace(ctx), Metal.threadgroup_position_in_grid_1d(), Metal.thread_position_in_threadgroup_1d())
        return I in __ndrange(ctx)
    else
        return true
    end
end

import KernelAbstractions: groupsize, __groupsize, __workitems_iterspace, add_float_contract, sub_float_contract, mul_float_contract
import KernelAbstractions: SharedMemory, Scratchpad, __synchronize, __size

###
# GPU implementation of shared memory
###
@device_override @inline function SharedMemory(::Type{T}, ::Val{Dims}, ::Val{Id}) where {T, Dims, Id}
    ptr = Metal.emit_threadgroup_memory(T, Val(prod(Dims)))
    Metal.MtlDeviceArray(Dims, ptr)
end

###
# GPU implementation of scratch memory
# - private memory for each workitem
###

@device_override @inline function Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    StaticArrays.MArray{__size(Dims), T}(undef)
end

@device_override @inline function __synchronize()
    Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)
end

@device_override @inline function __print(args...)
    # TODO
end

KernelAbstractions.argconvert(::Kernel{MetalDevice}, arg) = Metal.mtlconvert(arg)

end
