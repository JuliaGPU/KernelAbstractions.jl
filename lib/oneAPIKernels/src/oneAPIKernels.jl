module oneAPIKernels

import oneAPI
import oneAPI: oneL0
import StaticArrays
import KernelAbstractions

export oneAPIDevice

KernelAbstractions.get_device(::Type{<:oneAPI.oneArray}) = oneAPIDevice()


const FREE_QUEUES = oneL0.ZeCommandQueue[]
const QUEUES = oneL0.ZeCommandQueue[]
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
    ctx = oneAPI.context()
    dev = oneAPI.device()
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

        queue = oneL0.ZeCommandQueue(ctx, dev)
        push!(QUEUES, queue)
        return queue
    end
end

function default_queue()
    ctx = oneAPI.context()
    dev = oneAPI.device()
    oneAPI.global_queue(ctx, dev)
end

import KernelAbstractions: Event, CPUEvent, NoneEvent, MultiEvent, CPU, GPU, isdone, failed

struct oneAPIDevice <: GPU end

struct oneAPIEvent <: Event
    event::oneL0.ZeEvent
end

failed(::oneAPIEvent) = false
isdone(ev::oneAPIEvent) = Base.isdone(ev.event)

function new_event(ctx=oneAPI.context(), dev=oneAPI.device())
    ctx = oneAPI.context()
    dev = oneAPI.device()
    event_pool = oneAPI.ZeEventPool(ctx, 1, dev)
    # XXX: this is inefficient
    event_pool[1]
end

# create an event for synchronizing against code that uses the task-local command queue
function Event(::oneAPIDevice)
    event = new_event()
    queue = default_queue()
    oneL0.execute!(queue) do list
        oneL0.append_signal!(list, event)
    end
    oneAPIEvent(event)
end

import Base: wait

wait(ev::oneAPIEvent, progress=nothing) = wait(CPU(), ev, progress)

wait(::CPU, ev::oneAPIEvent, progress=nothing) = wait(ev.event)
function wait(::oneAPIDevice, ev::oneAPIEvent, progress=nothing;
              queue::oneL0.ZeCommandQueue=default_queue())
    oneL0.execute!(queue) do list
        oneL0.append_wait!(list, ev.event)
    end
end
wait(::oneAPIDevice, ev::NoneEvent, progress=nothing; queue=nothing) = nothing

function wait(::oneAPIDevice, ev::MultiEvent, progress=nothing;
              queue::oneL0.ZeCommandQueue=default_queue())
    dependencies = collect(ev.events)
    onedeps  =  filter(d->d isa oneAPIEvent,    dependencies)
    otherdeps = filter(d->!(d isa oneAPIEvent), dependencies)
    oneL0.execute!(queue) do list
        for event in onedeps
            oneL0.append_wait!(list, event.event)
        end
    end
    for event in otherdeps
        wait(oneAPIDevice(), event, progress; queue)
    end
end

wait(::oneAPIDevice, ev::CPUEvent, progress=nothing; queue=nothing) =
    error("GPU->CPU wait not implemented")

function KernelAbstractions.async_copy!(::oneAPIDevice, A, B; dependencies=nothing, progress=nothing)
    queue = next_queue()
    wait(oneAPIDevice(), MultiEvent(dependencies), progress; queue)
    # XXX: append_copy! can also wait for events

    event = new_event()

    oneL0.execute!(queue) do list
        destptr = pointer(A)
        srcptr  = pointer(B)
        nb      = sizeof(A)
        oneL0.append_copy!(list, destptr, srcptr, nb, event)
    end

   oneAPIEvent(event)
end

import KernelAbstractions: Kernel, StaticSize, DynamicSize, partition, blocks, workitems, launch_config

###
# Kernel launch
###
function launch_config(kernel::Kernel{oneAPIDevice}, ndrange, workgroupsize)
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

function (obj::Kernel{oneAPIDevice})(args...; ndrange=nothing, dependencies=nothing,
                                     workgroupsize=nothing, progress=nothing)
    ndrange, workgroupsize, iterspace, dynamic = launch_config(obj, ndrange, workgroupsize)
    # this might not be the final context, since we may tune the workgroupsize
    ctx = mkcontext(obj, ndrange, iterspace)
    kernel = oneAPI.@oneapi launch=false obj.f(ctx, args...)

    # figure out the optimal workgroupsize automatically
    if KernelAbstractions.workgroupsize(obj) <: DynamicSize && workgroupsize === nothing
        items = oneAPI.suggest_groupsize(kernel.fun, prod(ndrange)).x
        # XXX: the z dimension of the suggested group size is often non-zero. use this?
        workgroupsize = threads_to_workgroupsize(items, ndrange)
        iterspace, dynamic = partition(obj, ndrange, workgroupsize)
        ctx = mkcontext(obj, ndrange, iterspace)
    end

    nblocks = length(blocks(iterspace))
    threads = length(workitems(iterspace))

    if nblocks == 0
        return MultiEvent(dependencies)
    end

    queue = next_queue()
    wait(oneAPIDevice(), MultiEvent(dependencies), progress; queue)

    event = new_event()

    # Launch kernel
    kernel(ctx, args...; items=threads, groups=nblocks, queue=queue)

    oneL0.execute!(queue) do list
        oneL0.append_signal!(list, event)
    end
    # XXX: append_launch! can also wait and signal events

    return oneAPIEvent(event)
end

import oneAPI: @device_override

import KernelAbstractions: CompilerMetadata, DynamicCheck, LinearIndices
import KernelAbstractions: __index_Local_Linear, __index_Group_Linear, __index_Global_Linear, __index_Local_Cartesian, __index_Group_Cartesian, __index_Global_Cartesian, __validindex, __print
import KernelAbstractions: mkcontext, expand, __iterspace, __ndrange, __dynamic_checkbounds

function mkcontext(kernel::Kernel{oneAPIDevice}, _ndrange, iterspace)
    metadata = CompilerMetadata{KernelAbstractions.ndrange(kernel), DynamicCheck}(_ndrange, iterspace)
end
function mkcontext(kernel::Kernel{oneAPIDevice}, I, _ndrange, iterspace, ::Dynamic) where Dynamic
    metadata = CompilerMetadata{KernelAbstractions.ndrange(kernel), Dynamic}(I, _ndrange, iterspace)
end

@device_override @inline function __index_Local_Linear(ctx)
    return oneAPI.get_local_id(0)
end

@device_override @inline function __index_Group_Linear(ctx)
    return oneAPI.get_group_id(0)
end

@device_override @inline function __index_Global_Linear(ctx)
    I =  @inbounds expand(__iterspace(ctx), oneAPI.get_group_id(0), oneAPI.get_local_id(0))
    # TODO: This is unfortunate, can we get the linear index cheaper
    @inbounds LinearIndices(__ndrange(ctx))[I]
end

@device_override @inline function __index_Local_Cartesian(ctx)
    @inbounds workitems(__iterspace(ctx))[oneAPI.get_local_id(0)]
end

@device_override @inline function __index_Group_Cartesian(ctx)
    @inbounds blocks(__iterspace(ctx))[oneAPI.get_group_id(0)]
end

@device_override @inline function __index_Global_Cartesian(ctx)
    return @inbounds expand(__iterspace(ctx), oneAPI.get_group_id(0), oneAPI.get_local_id(0))
end

@device_override @inline function __validindex(ctx)
    if __dynamic_checkbounds(ctx)
        I = @inbounds expand(__iterspace(ctx), oneAPI.get_group_id(0), oneAPI.get_local_id(0))
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
    ptr = oneAPI.emit_localmemory(T, Val(prod(Dims)))
    oneAPI.oneDeviceArray(Dims, ptr)
end

###
# GPU implementation of scratch memory
# - private memory for each workitem
###

@device_override @inline function Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    StaticArrays.MArray{__size(Dims), T}(undef)
end

@device_override @inline function __synchronize()
    oneAPI.barrier()
end

@device_override @inline function __print(args...)
    oneAPI._print(args...)
end

KernelAbstractions.argconvert(::Kernel{oneAPIDevice}, arg) = oneAPI.kernel_convert(arg)

end
