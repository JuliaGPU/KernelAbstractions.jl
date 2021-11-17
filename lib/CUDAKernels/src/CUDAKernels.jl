module CUDAKernels

import CUDA
import SpecialFunctions
import StaticArrays
import StaticArrays: MArray
import Cassette
import Adapt
import KernelAbstractions

export CUDADevice

const FREE_STREAMS = CUDA.CuStream[]
const STREAMS = CUDA.CuStream[]
const STREAM_GC_THRESHOLD = Ref{Int}(16)

# This code is loaded after an `@init` step
if haskey(ENV, "KERNELABSTRACTIONS_STREAMS_GC_THRESHOLD")
    global STREAM_GC_THRESHOLD[] = parse(Int, ENV["KERNELABSTRACTIONS_STREAMS_GC_THRESHOLD"])
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
const STREAM_GC_LOCK = Threads.ReentrantLock()
function next_stream()
    lock(STREAM_GC_LOCK) do
        if !isempty(FREE_STREAMS)
            return pop!(FREE_STREAMS)
        end

        if length(STREAMS) > STREAM_GC_THRESHOLD[]
            for stream in STREAMS
                if CUDA.query(stream)
                    push!(FREE_STREAMS, stream)
                end
            end
        end

        if !isempty(FREE_STREAMS)
            return pop!(FREE_STREAMS)
        end
        stream = CUDA.CuStream(flags = CUDA.STREAM_NON_BLOCKING)
        push!(STREAMS, stream)
        return stream
    end
end

import KernelAbstractions: Event, CPUEvent, NoneEvent, MultiEvent, CPU, GPU, isdone, failed

struct CUDADevice <: GPU end

struct CudaEvent <: Event
    event::CUDA.CuEvent
end

failed(::CudaEvent) = false
isdone(ev::CudaEvent) = CUDA.query(ev.event)

function Event(::CUDADevice)
    stream = CUDA.stream()
    event = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    CUDA.record(event, stream)
    CudaEvent(event)
end

import Base: wait

wait(ev::CudaEvent, progress=yield) = wait(CPU(), ev, progress)

function wait(::CPU, ev::CudaEvent, progress=nothing)
    isdone(ev) && return nothing

    event = Base.Threads.Event()
    stream = next_stream()
    wait(CUDADevice(), ev, nothing, stream)
    CUDA.launch(;stream) do
        notify(event)
    end
    wait(event)
end

# Use this to synchronize between computation using the task local stream
wait(::CUDADevice, ev::CudaEvent, progress=nothing, stream=CUDA.stream()) = CUDA.wait(ev.event, stream)
wait(::CUDADevice, ev::NoneEvent, progress=nothing, stream=nothing) = nothing

function wait(::CUDADevice, ev::MultiEvent, progress=nothing, stream=CUDA.stream())
    dependencies = collect(ev.events)
    cudadeps  = filter(d->d isa CudaEvent,    dependencies)
    otherdeps = filter(d->!(d isa CudaEvent), dependencies)
    for event in cudadeps
        CUDA.wait(event.event, stream)
    end
    for event in otherdeps
        wait(CUDADevice(), event, progress, stream)
    end
end

function wait(::CUDADevice, ev::CPUEvent, progress=nothing, stream=nothing)
    error("""
    Waiting on the GPU for an CPU event to finish is currently not supported.
    We have encountered deadlocks arising, due to interactions with the CUDA
    driver.
    """)
end

###
# async_copy
###
# - IdDict does not free the memory
# - WeakRef dict does not unique the key by objectid
const __pinned_memory = Dict{UInt64, WeakRef}()

function __pin!(a)
    # use pointer instead of objectid?
    oid = objectid(a)
    if haskey(__pinned_memory, oid) && __pinned_memory[oid].value !== nothing
        return nothing
    end
    ad = CUDA.Mem.register(CUDA.Mem.Host, pointer(a), sizeof(a))
    finalizer(_ -> CUDA.Mem.unregister(ad), a)
    __pinned_memory[oid] = WeakRef(a)
    return nothing
end

function KernelAbstractions.async_copy!(::CUDADevice, A, B; dependencies=nothing, progress=yield)
    A isa Array && __pin!(A)
    B isa Array && __pin!(B)

    stream = next_stream()
    wait(CUDADevice(), MultiEvent(dependencies), progress, stream)
    event = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    GC.@preserve A B begin
        destptr = pointer(A)
        srcptr  = pointer(B)
        N       = length(A)
        unsafe_copyto!(destptr, srcptr, N, async=true, stream=stream)
    end

    CUDA.record(event, stream)
    return CudaEvent(event)
end

import KernelAbstractions: Kernel, StaticSize, DynamicSize, partition, blocks, workitems, launch_config

###
# Kernel launch
###
function launch_config(kernel::Kernel{CUDADevice}, ndrange, workgroupsize)
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

function (obj::Kernel{CUDADevice})(args...; ndrange=nothing, dependencies=Event(CUDADevice()), workgroupsize=nothing, progress=yield)

    ndrange, workgroupsize, iterspace, dynamic = launch_config(obj, ndrange, workgroupsize)
    # this might not be the final context, since we may tune the workgroupsize
    ctx = mkcontext(obj, ndrange, iterspace)
    kernel = CUDA.@cuda launch=false name=String(nameof(obj.f)) Cassette.overdub(CUDACTX, obj.f, ctx, args...)

    # figure out the optimal workgroupsize automatically
    if KernelAbstractions.workgroupsize(obj) <: DynamicSize && workgroupsize === nothing
        config = CUDA.launch_configuration(kernel.fun; max_threads=prod(ndrange))
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

    nblocks = length(blocks(iterspace))
    threads = length(workitems(iterspace))

    if nblocks == 0
        return MultiEvent(dependencies)
    end

    stream = next_stream()
    wait(CUDADevice(), MultiEvent(dependencies), progress, stream)

    # Launch kernel
    event = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    kernel(CUDACTX, obj.f, ctx, args...; threads=threads, blocks=nblocks, stream=stream)

    CUDA.record(event, stream)
    return CudaEvent(event)
end

Cassette.@context CUDACtx

import KernelAbstractions: CompilerMetadata, CompilerPass, DynamicCheck, LinearIndices
import KernelAbstractions: __index_Local_Linear, __index_Group_Linear, __index_Global_Linear, __index_Local_Cartesian, __index_Group_Cartesian, __index_Global_Cartesian, __validindex, __print
import KernelAbstractions: mkcontext, expand, __iterspace, __ndrange, __dynamic_checkbounds

const CUDACTX = Cassette.disablehooks(CUDACtx(pass = CompilerPass))
KernelAbstractions.cassette(::Kernel{CUDADevice}) = CUDACTX

function mkcontext(kernel::Kernel{CUDADevice}, _ndrange, iterspace)
    CompilerMetadata{KernelAbstractions.ndrange(kernel), DynamicCheck}(_ndrange, iterspace)
end

@inline function Cassette.overdub(::CUDACtx, ::typeof(__index_Local_Linear), ctx)
    return CUDA.threadIdx().x
end

@inline function Cassette.overdub(::CUDACtx, ::typeof(__index_Group_Linear), ctx)
    return CUDA.blockIdx().x
end

@inline function Cassette.overdub(::CUDACtx, ::typeof(__index_Global_Linear), ctx)
    I =  @inbounds expand(__iterspace(ctx), CUDA.blockIdx().x, CUDA.threadIdx().x)
    # TODO: This is unfortunate, can we get the linear index cheaper
    @inbounds LinearIndices(__ndrange(ctx))[I]
end

@inline function Cassette.overdub(::CUDACtx, ::typeof(__index_Local_Cartesian), ctx)
    @inbounds workitems(__iterspace(ctx))[CUDA.threadIdx().x]
end

@inline function Cassette.overdub(::CUDACtx, ::typeof(__index_Group_Cartesian), ctx)
    @inbounds blocks(__iterspace(ctx))[CUDA.blockIdx().x]
end

@inline function Cassette.overdub(::CUDACtx, ::typeof(__index_Global_Cartesian), ctx)
    return @inbounds expand(__iterspace(ctx), CUDA.blockIdx().x, CUDA.threadIdx().x)
end

@inline function Cassette.overdub(::CUDACtx, ::typeof(__validindex), ctx)
    if __dynamic_checkbounds(ctx)
        I = @inbounds expand(__iterspace(ctx), CUDA.blockIdx().x, CUDA.threadIdx().x)
        return I in __ndrange(ctx)
    else
        return true
    end
end

import KernelAbstractions: groupsize, __groupsize, __workitems_iterspace, add_float_contract, sub_float_contract, mul_float_contract

KernelAbstractions.generate_overdubs(@__MODULE__, CUDACtx)

###
# CUDA specific method rewrites
###

@inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Float64, y::Float64) = ^(x, y)
@inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Float32, y::Float32) = ^(x, y)
@inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Float64, y::Int32)   = ^(x, y)
@inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Float32, y::Int32)   = ^(x, y)
@inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Union{Float32, Float64}, y::Int64) = ^(x, y)

# libdevice.jl
const cudafuns = (:cos, :cospi, :sin, :sinpi, :tan,
          :acos, :asin, :atan,
          :cosh, :sinh, :tanh,
          :acosh, :asinh, :atanh,
          :log, :log10, :log1p, :log2,
          :exp, :exp2, :exp10, :expm1, :ldexp,
          # :isfinite, :isinf, :isnan, :signbit,
          :abs,
          :sqrt, :cbrt,
          :ceil, :floor,)
for f in cudafuns
    @eval function Cassette.overdub(ctx::CUDACtx, ::typeof(Base.$f), x::Union{Float32, Float64})
        @Base._inline_meta
        return Base.$f(x)
    end
end

@inline Cassette.overdub(::CUDACtx, ::typeof(sincos), x::Union{Float32, Float64}) = (Base.sin(x), Base.cos(x))
@inline Cassette.overdub(::CUDACtx, ::typeof(exp), x::Union{ComplexF32, ComplexF64}) = Base.exp(x)

@inline Cassette.overdub(::CUDACtx, ::typeof(SpecialFunctions.gamma), x::Union{Float32, Float64}) = CUDA.tgamma(x)
@inline Cassette.overdub(::CUDACtx, ::typeof(SpecialFunctions.erf), x::Union{Float32, Float64}) = SpecialFunctions.erf(x)
@inline Cassette.overdub(::CUDACtx, ::typeof(SpecialFunctions.erfc), x::Union{Float32, Float64}) = SpecialFunctions.erfc(x)
@inline Cassette.overdub(::CUDACtx, ::typeof(SpecialFunctions.erfcx), x::Union{Float32, Float64}) = SpecialFunctions.erfcx(x)

@static if Base.isbindingresolved(CUDA, :emit_shmem) && Base.isdefined(CUDA, :emit_shmem)
    const emit_shmem = CUDA.emit_shmem
else
    const emit_shmem = CUDA._shmem
end

import KernelAbstractions: ConstAdaptor, SharedMemory, Scratchpad, __synchronize, __size

###
# GPU implementation of shared memory
###

@inline function Cassette.overdub(::CUDACtx, ::typeof(SharedMemory), ::Type{T}, ::Val{Dims}, ::Val{Id}) where {T, Dims, Id}
    ptr = emit_shmem(T, Val(prod(Dims)))
    CUDA.CuDeviceArray(Dims, ptr)
end

###
# GPU implementation of scratch memory
# - private memory for each workitem
###

@inline function Cassette.overdub(::CUDACtx, ::typeof(Scratchpad), ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    MArray{__size(Dims), T}(undef)
end

@inline function Cassette.overdub(::CUDACtx, ::typeof(__synchronize))
    CUDA.sync_threads()
end

@inline function Cassette.overdub(::CUDACtx, ::typeof(__print), args...)
    CUDA._cuprint(args...)
end

###
# GPU implementation of const memory
###

Adapt.adapt_storage(to::ConstAdaptor, a::CUDA.CuDeviceArray) = Base.Experimental.Const(a)

# Argument conversion
KernelAbstractions.argconvert(k::Kernel{CUDADevice}, arg) = CUDA.cudaconvert(arg)

end
