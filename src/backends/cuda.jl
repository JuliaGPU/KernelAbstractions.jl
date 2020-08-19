import CUDA
import SpecialFunctions

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
# - Add locking
function next_stream()
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

struct CudaEvent <: Event
    event::CUDA.CuEvent
end

failed(::CudaEvent) = false
isdone(ev::CudaEvent) = CUDA.query(ev.event)

function Event(::CUDADevice)
    stream = CUDA.CuDefaultStream()
    event = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    CUDA.record(event, stream)
    CudaEvent(event)
end

wait(ev::CudaEvent, progress=yield) = wait(CPU(), ev, progress)

function wait(::CPU, ev::CudaEvent, progress=yield)
    if progress === nothing
        CUDA.synchronize(ev.event)
    else
        while !isdone(ev)
            progress()
        end
    end
end

# Use this to synchronize between computation using the CuDefaultStream
wait(::CUDADevice, ev::CudaEvent, progress=nothing, stream=CUDA.CuDefaultStream()) = CUDA.wait(ev.event, stream)
wait(::CUDADevice, ev::NoneEvent, progress=nothing, stream=nothing) = nothing

function wait(::CUDADevice, ev::MultiEvent, progress=nothing, stream=CUDA.CuDefaultStream())
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

include("cusynchronization.jl")
import .CuSynchronization: unsafe_volatile_load, unsafe_volatile_store!

function wait(::CUDADevice, ev::CPUEvent, progress=nothing, stream=nothing)
    error("""
    Waiting on the GPU for an CPU event to finish is currently not supported.
    We have encountered deadlocks arising, due to interactions with the CUDA
    driver. If you are certain that you are deadlock free, you can use `unsafe_wait`
    instead.
    """)
end

# This implements waiting for a CPUEvent on the GPU.
# Most importantly this implementation needs to be asynchronous w.r.t to the host,
# otherwise one could introduce deadlocks with outside event systems.
# It uses a device visible host buffer to create a barrier/semaphore.
# On a CPU task we wait for the `ev` to finish and then signal the GPU
# by setting the flag 0->1, the CPU then in return needs to wait for the GPU
# to set trhe flag 1->2 so that we can deallocate the memory.
# TODO:
# - In case of an error we should probably also kill the waiting GPU code.
unsafe_wait(dev::Device, ev, progress=nothing) = wait(dev, ev, progress) 
function unsafe_wait(::CUDADevice, ev::CPUEvent, progress=nothing, stream=CUDA.CuDefaultStream())
    buf = CUDA.Mem.alloc(CUDA.Mem.HostBuffer, sizeof(UInt32), CUDA.Mem.HOSTREGISTER_DEVICEMAP)
    unsafe_store!(convert(Ptr{UInt32}, buf), UInt32(0))
    # TODO: Switch to `@spawn` when CUDA.jl is thread-safe
    @async begin
        try
            wait(ev.task)
        catch err
            bt = catch_backtrace()
            @error "Error thrown during CUDA wait on CPUEvent" _ex=(err, bt)
        finally
            @debug "notifying GPU"
            unsafe_volatile_store!(convert(Ptr{UInt32}, buf), UInt32(1))
            while !(unsafe_volatile_load(convert(Ptr{UInt32}, buf)) == UInt32(2))
                yield()
            end
            @debug "GPU released"
            CUDA.Mem.free(buf)
        end
    end
    ptr = convert(CUDA.DevicePtr{UInt32}, convert(CUDA.Mem.CuPtr{UInt32}, buf))
    sem = CuSynchronization.Semaphore(ptr, UInt32(1))
    CUDA.@cuda threads=1 stream=stream CuSynchronization.wait(sem)
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

function async_copy!(::CUDADevice, A, B; dependencies=nothing, progress=yield)
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



###
# Kernel launch
###
function (obj::Kernel{CUDADevice})(args...; ndrange=nothing, dependencies=nothing, workgroupsize=nothing, progress=yield)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize, )
    end

    if KernelAbstractions.workgroupsize(obj) <: DynamicSize && workgroupsize === nothing
        # TODO: allow for NDRange{1, DynamicSize, DynamicSize}(nothing, nothing)
        #       and actually use CUDA autotuning
        workgroupsize = (256,)
    end
    # If the kernel is statically sized we can tell the compiler about that
    if KernelAbstractions.workgroupsize(obj) <: StaticSize
        maxthreads = prod(get(KernelAbstractions.workgroupsize(obj)))
    else
        maxthreads = nothing
    end

    iterspace, dynamic = partition(obj, ndrange, workgroupsize)

    nblocks = length(blocks(iterspace))
    threads = length(workitems(iterspace))

    if nblocks == 0
        return MultiEvent(dependencies)
    end

    stream = next_stream()
    wait(CUDADevice(), MultiEvent(dependencies), progress, stream)

    ctx = mkcontext(obj, ndrange, iterspace)
    # Launch kernel
    event = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    CUDA.@cuda(threads=threads, blocks=nblocks, stream=stream,
               name=String(nameof(obj.f)), maxthreads=maxthreads,
               Cassette.overdub(ctx, obj.f, args...))

    CUDA.record(event, stream)
    return CudaEvent(event)
end

Cassette.@context CUDACtx

function mkcontext(kernel::Kernel{CUDADevice}, _ndrange, iterspace)
    metadata = CompilerMetadata{ndrange(kernel), DynamicCheck}(_ndrange, iterspace)
    Cassette.disablehooks(CUDACtx(pass = CompilerPass, metadata=metadata))
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__index_Local_Linear))
    return CUDA.threadIdx().x
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__index_Group_Linear))
    return CUDA.blockIdx().x
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__index_Global_Linear))
    I =  @inbounds expand(__iterspace(ctx.metadata), CUDA.blockIdx().x, CUDA.threadIdx().x)
    # TODO: This is unfortunate, can we get the linear index cheaper
    @inbounds LinearIndices(__ndrange(ctx.metadata))[I]
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__index_Local_Cartesian))
    @inbounds workitems(__iterspace(ctx.metadata))[CUDA.threadIdx().x]
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__index_Group_Cartesian))
    @inbounds blocks(__iterspace(ctx.metadata))[CUDA.blockIdx().x]
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__index_Global_Cartesian))
    return @inbounds expand(__iterspace(ctx.metadata), CUDA.blockIdx().x, CUDA.threadIdx().x)
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__validindex))
    if __dynamic_checkbounds(ctx.metadata)
        I = @inbounds expand(__iterspace(ctx.metadata), CUDA.blockIdx().x, CUDA.threadIdx().x)
        return I in __ndrange(ctx.metadata)
    else
        return true
    end
end

generate_overdubs(CUDACtx)

###
# CUDA specific method rewrites
###

@inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Float64, y::Float64) = CUDA.pow(x, y)
@inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Float32, y::Float32) = CUDA.pow(x, y)
@inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Float64, y::Int32)   = CUDA.pow(x, y)
@inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Float32, y::Int32)   = CUDA.pow(x, y)
@inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Union{Float32, Float64}, y::Int64) = CUDA.pow(x, y)

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
        return CUDA.$f(x)
    end
end

@inline Cassette.overdub(::CUDACtx, ::typeof(sincos), x::Union{Float32, Float64}) = (CUDA.sin(x), CUDA.cos(x))
@inline Cassette.overdub(::CUDACtx, ::typeof(exp), x::Union{ComplexF32, ComplexF64}) = CUDA.exp(x)

@inline Cassette.overdub(::CUDACtx, ::typeof(SpecialFunctions.gamma), x::Union{Float32, Float64}) = CUDA.tgamma(x)

@static if Base.isbindingresolved(CUDA, :emit_shmem) && Base.isdefined(CUDA, :emit_shmem)
    const emit_shmem = CUDA.emit_shmem
else
    const emit_shmem = CUDA._shmem
end

ptr = CUDA._shmem(Val(Id), T, Val(prod(Dims)))
###
# GPU implementation of shared memory
###
@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(SharedMemory), ::Type{T}, ::Val{Dims}, ::Val{Id}) where {T, Dims, Id}
    ptr = emit_shmem(Val(Id), T, Val(prod(Dims)))
    CUDA.CuDeviceArray(Dims, ptr)
end

###
# GPU implementation of scratch memory
# - private memory for each workitem
###

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(Scratchpad), ::Type{T}, ::Val{Dims}) where {T, Dims}
    MArray{__size(Dims), T}(undef)
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__synchronize))
    CUDA.sync_threads()
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__print), args...)
    CUDA._cuprint(args...)
end

###
# GPU implementation of `@Const`
###
struct ConstCuDeviceArray{T,N,A} <: AbstractArray{T,N}
    shape::Dims{N}
    ptr::CUDA.DevicePtr{T,A}

    # inner constructors, fully parameterized, exact types (ie. Int not <:Integer)
    ConstCuDeviceArray{T,N,A}(shape::Dims{N}, ptr::CUDA.DevicePtr{T,A}) where {T,A,N} = new(shape,ptr)
end

Adapt.adapt_storage(to::ConstAdaptor, a::CUDA.CuDeviceArray{T,N,A}) where {T,N,A} = ConstCuDeviceArray{T, N, A}(a.shape, a.ptr)

Base.pointer(a::ConstCuDeviceArray) = a.ptr
Base.pointer(a::ConstCuDeviceArray, i::Integer) =
    pointer(a) + (i - 1) * Base.elsize(a)

Base.elsize(::Type{<:ConstCuDeviceArray{T}}) where {T} = sizeof(T)
Base.size(g::ConstCuDeviceArray) = g.shape
Base.length(g::ConstCuDeviceArray) = prod(g.shape)
Base.IndexStyle(::Type{<:ConstCuDeviceArray}) = Base.IndexLinear()

Base.unsafe_convert(::Type{CUDA.DevicePtr{T,A}}, a::ConstCuDeviceArray{T,N,A}) where {T,A,N} = pointer(a)

@inline function Base.getindex(A::ConstCuDeviceArray{T}, index::Integer) where {T}
    @boundscheck checkbounds(A, index)
    align = Base.datatype_alignment(T)
    CUDA.unsafe_cached_load(pointer(A), index, Val(align))::T
end

@inline function Base.unsafe_view(arr::ConstCuDeviceArray{T, 1, A}, I::Vararg{Base.ViewIndex,1}) where {T, A}
    ptr = pointer(arr) + (I[1].start-1)*sizeof(T)
    len = I[1].stop - I[1].start + 1
    return ConstCuDeviceArray{T,1,A}(len, ptr)
end
