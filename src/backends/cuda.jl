import CUDAnative, CUDAdrv
import CUDAnative: cufunction, DevicePtr
import CUDAdrv: CuEvent, CuStream, CuDefaultStream

const FREE_STREAMS = CuStream[]
const STREAMS = CuStream[]
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
            if CUDAdrv.query(stream)
                push!(FREE_STREAMS, stream)
            end
        end
    end

    if !isempty(FREE_STREAMS)
        return pop!(FREE_STREAMS)
    end

    stream = CUDAdrv.CuStream(CUDAdrv.STREAM_NON_BLOCKING)
    push!(STREAMS, stream)
    return stream
end

struct CudaEvent <: Event
    event::CuEvent
end
function wait(ev::CudaEvent, progress=nothing)
    if progress === nothing
        CUDAdrv.wait(ev.event)
    else
        while !CUDAdrv.query(ev.event)
            progress()
            # do we need to `yield` here?
        end
    end
end

function (obj::Kernel{CUDA})(args...; ndrange=nothing, dependencies=nothing, workgroupsize=nothing)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize, )
    end
    if dependencies isa Event
        dependencies = (dependencies,)
    end

    stream = next_stream()
    if dependencies !== nothing
        for event in dependencies
            @assert event isa CudaEvent
            CUDAdrv.wait(event.event, stream)
        end
    end

    if KernelAbstractions.workgroupsize(obj) <: DynamicSize && workgroupsize === nothing
        # TODO: allow for NDRange{1, DynamicSize, DynamicSize}(nothing, nothing)
        #       and actually use CUDAnative autotuning
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

    ctx = mkcontext(obj, ndrange, iterspace)
    # Launch kernel
    event = CuEvent(CUDAdrv.EVENT_DISABLE_TIMING)
    CUDAnative.@cuda(threads=threads, blocks=nblocks, stream=stream,
                     name=String(nameof(obj.f)), maxthreads=maxthreads,
                     Cassette.overdub(ctx, obj.f, args...))

    CUDAdrv.record(event, stream)
    return CudaEvent(event)
end

Cassette.@context CUDACtx

function mkcontext(kernel::Kernel{CUDA}, _ndrange, iterspace)
    metadata = CompilerMetadata{ndrange(kernel), true}(_ndrange, iterspace)
    Cassette.disablehooks(CUDACtx(pass = CompilerPass, metadata=metadata))
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__index_Local_Linear))
    return CUDAnative.threadIdx().x
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__index_Global_Linear))
    I =  @inbounds expand(__iterspace(ctx.metadata), CUDAnative.blockIdx().x, CUDAnative.threadIdx().x)
    # TODO: This is unfortunate, can we get the linear index cheaper
    @inbounds LinearIndices(__ndrange(ctx.metadata))[I]
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__index_Local_Cartesian))
    @inbounds workitems(__iterspace(ctx.metadata))[CUDAnative.threadIdx().x]
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__index_Global_Cartesian))
    return @inbounds expand(__iterspace(ctx.metadata), CUDAnative.blockIdx().x, CUDAnative.threadIdx().x)
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__validindex))
    if __dynamic_checkbounds(ctx.metadata)
        I = @inbounds expand(__iterspace(ctx.metadata), CUDAnative.blockIdx().x, CUDAnative.threadIdx().x)
        return I in __ndrange(ctx.metadata)
    else
        return true
    end
end

generate_overdubs(CUDACtx)

###
# CUDA specific method rewrites
###

@inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Float64, y::Float64) = CUDAnative.pow(x, y)
@inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Float32, y::Float32) = CUDAnative.pow(x, y)
@inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Float64, y::Int32)   = CUDAnative.pow(x, y)
@inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Float32, y::Int32)   = CUDAnative.pow(x, y)
@inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Union{Float32, Float64}, y::Int64) = CUDAnative.pow(x, y)

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
        return CUDAnative.$f(x)
    end
end


###
# GPU implementation of shared memory
###
@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(SharedMemory), ::Type{T}, ::Val{Dims}, ::Val{Id}) where {T, Dims, Id}
    ptr = CUDAnative._shmem(Val(Id), T, Val(prod(Dims)))
    CUDAnative.CuDeviceArray(Dims, ptr)
end

###
# GPU implementation of scratch memory
# - private memory for each workitem
###

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(Scratchpad), ::Type{T}, ::Val{Dims}) where {T, Dims}
    MArray{__size(Dims), T}(undef)
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__synchronize))
    CUDAnative.sync_threads()
end

###
# GPU implementation of `@Const`
###
struct ConstCuDeviceArray{T,N,A} <: AbstractArray{T,N}
    shape::Dims{N}
    ptr::DevicePtr{T,A}

    # inner constructors, fully parameterized, exact types (ie. Int not <:Integer)
    ConstCuDeviceArray{T,N,A}(shape::Dims{N}, ptr::DevicePtr{T,A}) where {T,A,N} = new(shape,ptr)
end

Adapt.adapt_storage(to::ConstAdaptor, a::CUDAnative.CuDeviceArray{T,N,A}) where {T,N,A} = ConstCuDeviceArray{T, N, A}(a.shape, a.ptr)

Base.pointer(a::ConstCuDeviceArray) = a.ptr
Base.pointer(a::ConstCuDeviceArray, i::Integer) =
    pointer(a) + (i - 1) * Base.elsize(a)

Base.elsize(::Type{<:ConstCuDeviceArray{T}}) where {T} = sizeof(T)
Base.size(g::ConstCuDeviceArray) = g.shape
Base.length(g::ConstCuDeviceArray) = prod(g.shape)

Base.unsafe_convert(::Type{DevicePtr{T,A}}, a::ConstCuDeviceArray{T,N,A}) where {T,A,N} = pointer(a)

@inline function Base.getindex(A::ConstCuDeviceArray{T}, index::Integer) where {T}
    @boundscheck checkbounds(A, index)
    align = Base.datatype_alignment(T)
    CUDAnative.unsafe_cached_load(pointer(A), index, Val(align))::T
end
