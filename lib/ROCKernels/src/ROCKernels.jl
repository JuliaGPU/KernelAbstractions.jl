module ROCKernels

import AMDGPU
import AMDGPU: rocfunction, HSAAgent, HSAQueue, HSASignal, HSAStatusSignal, Mem
import StaticArrays
import StaticArrays: MArray
import Adapt
import KernelAbstractions

export ROCDevice

get_computing_device(::AMDGPU.ROCArray) = ROCDevice()


const FREE_QUEUES = HSAQueue[]
const QUEUES = HSAQueue[]
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

        # FIXME: Which agent?
        queue = HSAQueue()
        push!(QUEUES, queue)
        return queue
    end
end

import KernelAbstractions: Event, CPUEvent, NoneEvent, MultiEvent, CPU, GPU, isdone, failed

struct ROCEvent{T<:Union{AMDGPU.HSA.Signal,HSAStatusSignal}} <: Event
    event::T
end
ROCEvent(signal::HSASignal) = ROCEvent(signal.signal[])

cpuwait(ev::ROCEvent{AMDGPU.HSA.Signal}) = wait(HSASignal(ev.event))
cpuwait(ev::ROCEvent{HSAStatusSignal}) = wait(ev.event)
gpuwait(ev::ROCEvent{AMDGPU.HSA.Signal}) = AMDGPU.device_signal_wait(ev.event, 0)
gpuwait(ev::ROCEvent{HSAStatusSignal}) = wait(ev.event)

Adapt.adapt_storage(::AMDGPU.Adaptor, ev::ROCEvent{HSAStatusSignal}) =
    ROCEvent(ev.event.signal.signal[])

failed(::ROCEvent) = false # FIXME
isdone(ev::ROCEvent) = true # FIXME

function Event(::ROCDevice)
    queue = AMDGPU.get_default_queue()
    # Returns an HSASignalSet containing signals
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

wait(::ROCDevice, ev::ROCEvent, progress=nothing, queue=nothing) = gpuwait(ev)
wait(::ROCDevice, ev::NoneEvent, progress=nothing, queue=nothing) = nothing

function wait(::ROCDevice, ev::MultiEvent, progress=nothing, queue=AMDGPU.get_default_queue())
    dependencies = collect(ev.events)
    rocdeps = map(d->d.event isa HSAStatusSignal ? d.event.signal.signal[] : d.event, filter(d->d isa ROCEvent, dependencies))
    wait.(rocdeps)
    #isempty(rocdeps) || wait(AMDGPU.barrier_and!(queue, rocdeps))
    for event in filter(d->!(d isa ROCEvent), dependencies)
        wait(ROCDevice(), event, progress)
    end
end

# TODO: Create a hostcall that waits on a specified Task
wait(::ROCDevice, ev::CPUEvent, progress=nothing, queue=nothing) = error("GPU->CPU wait not implemented")

function KernelAbstractions.async_copy!(::ROCDevice, A, B; dependencies=nothing, progress=nothing)
    # TODO: Register A and B
    wait(ROCDevice(), MultiEvent(dependencies), progress, AMDGPU.get_default_queue())
    GC.@preserve A B begin
        destptr = pointer(A)
        srcptr  = pointer(B)
        N       = length(A)
        # TODO: memory_copy_multiple, or custom copy kernel
        AMDGPU.HSA.memory_copy(destptr, srcptr, N) |> AMDGPU.check
    end

    return ROCEvent(HSASignal(0))
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
    threads = length(workitems(iterspace))

    if nblocks == 0
        return MultiEvent(dependencies)
    end

    queue = next_queue()
    wait(ROCDevice(), MultiEvent(dependencies), progress, queue)

    # Launch kernel
    event = AMDGPU.@roc(groupsize=threads, gridsize=nblocks*threads, queue=queue,
                        name=String(nameof(obj.f)), # TODO: maxthreads=maxthreads,
                        obj.f(ctx, args...))

    return ROCEvent(event.event)
end

import AMDGPU: @device_override

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
    return AMDGPU.threadIdx().x
end

@device_override @inline function __index_Group_Linear(ctx)
    return AMDGPU.blockIdx().x
end

@device_override @inline function __index_Global_Linear(ctx)
    I =  @inbounds expand(__iterspace(ctx), AMDGPU.blockIdx().x, AMDGPU.threadIdx().x)
    # TODO: This is unfortunate, can we get the linear index cheaper
    @inbounds LinearIndices(__ndrange(ctx))[I]
end

@device_override @inline function __index_Local_Cartesian(ctx)
    @inbounds workitems(__iterspace(ctx))[AMDGPU.threadIdx().x]
end

@device_override @inline function __index_Group_Cartesian(ctx)
    @inbounds blocks(__iterspace(ctx))[AMDGPU.blockIdx().x]
end

@device_override @inline function __index_Global_Cartesian(ctx)
    return @inbounds expand(__iterspace(ctx), AMDGPU.blockIdx().x, AMDGPU.threadIdx().x)
end

@device_override @inline function __validindex(ctx)
    if __dynamic_checkbounds(ctx)
        I = @inbounds expand(__iterspace(ctx), AMDGPU.blockIdx().x, AMDGPU.threadIdx().x)
        return I in __ndrange(ctx)
    else
        return true
    end
end

import KernelAbstractions: groupsize, __groupsize, __workitems_iterspace, add_float_contract, sub_float_contract, mul_float_contract
import KernelAbstractions: ConstAdaptor, SharedMemory, Scratchpad, __synchronize, __size

###
# GPU implementation of shared memory
###
@device_override @inline function SharedMemory(::Type{T}, ::Val{Dims}, ::Val{Id}) where {T, Dims, Id}
    ptr = AMDGPU.alloc_special(Val(Id), T, Val(AMDGPU.AS.Local), Val(prod(Dims)))
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
    AMDGPU.sync_workgroup()
end

@device_override @inline function __print(args...)
    for arg in args
        AMDGPU.@rocprintf("%s", arg)
    end
end

###
# GPU implementation of `@Const`
###
#=
struct ConstROCDeviceArray{T,N,A} <: AbstractArray{T,N}
    shape::Dims{N}
    ptr::Core.LLVMPtr{T,A}

    # inner constructors, fully parameterized, exact types (ie. Int not <:Integer)
    ConstROCDeviceArray{T,N,A}(shape::Dims{N}, ptr::Core.LLVMPtr{T,A}) where {T,A,N} = new(shape,ptr)
end

Adapt.adapt_storage(to::ConstAdaptor, a::AMDGPU.ROCDeviceArray{T,N,A}) where {T,N,A} = ConstROCDeviceArray{T, N, A}(a.shape, a.ptr)

Base.pointer(a::ConstROCDeviceArray) = a.ptr
Base.pointer(a::ConstROCDeviceArray, i::Integer) =
    pointer(a) + (i - 1) * Base.elsize(a)

Base.elsize(::Type{<:ConstROCDeviceArray{T}}) where {T} = sizeof(T)
Base.size(g::ConstROCDeviceArray) = g.shape
Base.length(g::ConstROCDeviceArray) = prod(g.shape)
Base.IndexStyle(::Type{<:ConstROCDeviceArray}) = Base.IndexLinear()

Base.unsafe_convert(::Type{Core.LLVMPtr{T,A}}, a::ConstROCDeviceArray{T,N,A}) where {T,A,N} = pointer(a)

@inline function Base.getindex(A::ConstROCDeviceArray{T}, index::Integer) where {T}
    @boundscheck checkbounds(A, index)
    align = Base.datatype_alignment(T)
    AMDGPU.unsafe_cached_load(pointer(A), index, Val(align))::T
end

@inline function Base.unsafe_view(arr::ConstROCDeviceArray{T, 1, A}, I::Vararg{Base.ViewIndex,1}) where {T, A}
    ptr = pointer(arr) + (I[1].start-1)*sizeof(T)
    len = I[1].stop - I[1].start + 1
    return ConstROCDeviceArray{T,1,A}(len, ptr)
end
=#

KernelAbstractions.argconvert(::Kernel{ROCDevice}, arg) = AMDGPU.rocconvert(arg)

end
