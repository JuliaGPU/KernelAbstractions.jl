import CUDAnative, CUDAdrv
import CUDAnative: cufunction
import CUDAdrv: CuEvent, CuStream, CuDefaultStream

STREAMS = CuStream[]
let id = 1
    global next_stream
    function next_stream()
        global id
        stream = STREAMS[id]
        if id < length(STREAMS)
            id += 1
        else
            id = 1
        end
    end
end

struct CudaEvent <: Event
    event::CuEvent
end
function wait(ev::CudaEvent)
    # TODO: MPI/libuv progress
    CUDAdrv.wait(ev.event)
end

@init begin
    if haskey(ENV, "KERNELABSTRACTIONS_STREAMS")
        nstreams = parse(Int, ENV["KERNELABSTRACTIONS_STREAMS"])
    else
        nstreams = 4
    end
    for i in 1:nstreams
        push!(STREAMS, CuStream(CUDAdrv.STREAM_NON_BLOCKING))
    end
end

function (obj::Kernel{CUDA})(args...; ndrange=nothing, dependencies=nothing, workgroupsize=nothing)
    if ndrange isa Int
        ndrange = (ndrange,)
    end
    if dependencies isa Event
        dependencies = (dependencies,)
    end

    # Be conservative and launch on CuDefaultStream
    if dependencies === nothing
        stream = CuDefaultStream()
    else
        stream = next_stream()
    end

    if dependencies !== nothing
        for event in dependencies
            @assert event isa CudaEvent
            CUDAdrv.wait(event.event, stream)
        end
    end

    event = CuEvent(CUDAdrv.EVENT_DISABLE_TIMING)

    # Launch kernel
    ctx = mkcontext(obj, ndrange)
    args = (ctx, obj.f, args...)
    GC.@preserve args begin
        kernel_args = map(CUDAnative.cudaconvert, args)
        kernel_tt = Tuple{map(Core.Typeof, kernel_args)...}

        # If the kernel is statically sized we can tell the compiler about that
        if KernelAbstractions.workgroupsize(obj) <: StaticSize 
            static_workgroupsize = get(KernelAbstractions.workgroupsize(obj))[1]
        else
            static_workgroupsize = nothing
        end

        kernel = CUDAnative.cufunction(Cassette.overdub, kernel_tt; name=String(nameof(obj.f)), maxthreads=static_workgroupsize)

        # Dynamically sized and size not prescribed, use autotuning
        if KernelAbstractions.workgroupsize(obj) <: DynamicSize && workgroupsize === nothing
            workgroupsize = CUDAnative.maxthreads(kernel)
        end

        if workgroupsize === nothing
            threads = static_workgroupsize
        else
            threads = workgroupsize
        end
        @assert threads !== nothing

        blocks, _ = partition(obj, ndrange, threads)
        kernel(kernel_args..., threads=threads, blocks=blocks, stream=stream)
    end
    CUDAdrv.record(event, stream)
    return CudaEvent(event)
end

Cassette.@context CUDACtx

function mkcontext(kernel::Kernel{CUDA}, _ndrange)
    metadata = CompilerMetadata{workgroupsize(kernel), ndrange(kernel), true}(_ndrange)
    Cassette.disablehooks(CUDACtx(pass = CompilerPass, metadata=metadata))
end


@inline function __gpu_groupsize(::CompilerMetadata{WorkgroupSize}) where {WorkgroupSize<:DynamicSize}
    CUDAnative.blockDim().x
end

@inline function __gpu_groupsize(cm::CompilerMetadata{WorkgroupSize}) where {WorkgroupSize<:StaticSize}
    __groupsize(cm)
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__index_Local_Linear))
    idx = CUDAnative.threadIdx().x
    return idx
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__index_Global_Linear))
    idx = CUDAnative.threadIdx().x
    workgroup = CUDAnative.blockIdx().x
    # XXX: have a verify mode where we check that our static dimensions are right
    #      e.g. that blockDim().x === __groupsize(ctx.metadata)
    return (workgroup - 1) * __gpu_groupsize(ctx.metadata) + idx
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__index_Local_Cartesian))
    error("@index(Local, Cartesian) is not yet defined")
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__index_Global_Cartesian))
    idx = CUDAnative.threadIdx().x
    workgroup = CUDAnative.blockIdx().x
    lI = (workgroup - 1) * __gpu_groupsize(ctx.metadata) + idx

    indices = __ndrange(ctx.metadata) 
    return @inbounds indices[lI]
end

@inline function Cassette.overdub(ctx::CUDACtx, ::typeof(__validindex))
    if __dynamic_checkbounds(ctx.metadata)
        idx = CUDAnative.threadIdx().x
        workgroup = CUDAnative.blockIdx().x
        lI = (workgroup - 1) * __gpu_groupsize(ctx.metadata) + idx
        maxidx = prod(size(__ndrange(ctx.metadata)))
        return lI <= maxidx
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
@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(SharedMemory), ::Type{T}, ::Val{Dims}, ::Val{Id}) where {T, Dims, Id}
    ptr = CUDAnative._shmem(Val(Id), T, Val(prod(Dims)))
    CUDAnative.CuDeviceArray(Dims, CUDAnative.DevicePtr{T, CUDAnative.AS.Shared}(ptr))
end

###
# GPU implementation of scratch memory
# - private memory for each workitem
###

@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(Scratchpad), ::Type{T}, ::Val{Dims}) where {T, Dims}
    MArray{__size(Dims), T}(undef)
end
