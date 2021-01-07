import SpecialFunctions

struct CPUEvent <: Event
    task::Core.Task
end
isdone(ev::CPUEvent) = Base.istaskdone(ev.task)
failed(ev::CPUEvent) = Base.istaskfailed(ev.task)

function Event(::CPU)
    return NoneEvent()
end

"""
    Event(f, args...; dependencies, progress, sticky=true)

Run function `f` with `args` in a Julia task. If `sticky` is `true` the task
is run on the thread that launched it.
"""
function Event(f, args...; dependencies=nothing, progress=nothing, sticky=true)
    T = Task() do
        wait(MultiEvent(dependencies), progress)
        f(args...)
    end
    T.sticky = sticky
    Base.schedule(T)
    return CPUEvent(T)
end

wait(ev::Union{CPUEvent, NoneEvent, MultiEvent}, progress=nothing) = wait(CPU(), ev, progress)
wait(::CPU, ev::NoneEvent, progress=nothing) = nothing

function wait(cpu::CPU, ev::MultiEvent, progress=nothing)
    events = ev.events
    N = length(events)
    alldone = ntuple(i->false, N)

    while !all(alldone)
        alldone = ntuple(N) do i
            if alldone[i]
                true
            else
                isdone(events[i])
            end
        end
        if progress === nothing
            yield()
        else
            progress()
        end
    end

    if any(failed, events)
        ex = CompositeException()
        for event in events
            if failed(event) && event isa CPUEvent
                push!(ex, Base.TaskFailedException(event.task))
            end
        end
        throw(ex)
    end
end

function wait(::CPU, ev::CPUEvent, progress=nothing)
    if progress === nothing
        wait(ev.task)
    else
        while !Base.istaskdone(ev.task)
            progress()
        end
        if Base.istaskfailed(ev.task)
            throw(Base.TaskFailedException(ev.task))
        end
    end
end

function async_copy!(::CPU, A, B; dependencies=nothing, progress=nothing)
    Event(copyto!, A, B, dependencies=dependencies, progress=progress)
end

function (obj::Kernel{CPU})(args...; ndrange=nothing, workgroupsize=nothing, dependencies=nothing, progress=nothing)
    ndrange, workgroupsize, iterspace, dynamic = launch_config(obj, ndrange, workgroupsize)

    if dependencies isa Event
        dependencies = (dependencies,)
    end

    if length(blocks(iterspace)) == 0
        return MultiEvent(dependencies)
    end

    Event(__run, obj, ndrange, iterspace, args, dynamic,
          dependencies=dependencies, progress=progress)
end

function launch_config(kernel::Kernel{CPU}, ndrange, workgroupsize)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize, )
    end

    if KernelAbstractions.workgroupsize(kernel) <: DynamicSize && workgroupsize === nothing
        workgroupsize = (1024,) # Vectorization, 4x unrolling, minimal grain size
    end
    iterspace, dynamic = partition(kernel, ndrange, workgroupsize)
    # partition checked that the ndrange's agreed
    if KernelAbstractions.ndrange(kernel) <: StaticSize
        ndrange = nothing
    end

    return ndrange, workgroupsize, iterspace, dynamic
end

# Inference barriers
function __run(obj, ndrange, iterspace, args, dynamic)
    N = length(iterspace)
    Nthreads = Threads.nthreads()
    if Nthreads == 1
        len, rem = N, 0
    else
        len, rem = divrem(N, Nthreads)
    end
    # not enough iterations for all the threads?
    if len == 0
        Nthreads = N
        len, rem = 1, 0
    end
    if Nthreads == 1
        __thread_run(1, len, rem, obj, ndrange, iterspace, args, dynamic)
    else
        @sync for tid in 1:Nthreads
            Threads.@spawn __thread_run(tid, len, rem, obj, ndrange, iterspace, args, dynamic)
        end
    end
    return nothing
end

function __thread_run(tid, len, rem, obj, ndrange, iterspace, args, dynamic)
    # compute this thread's iterations
    f = 1 + ((tid-1) * len)
    l = f + len - 1
    # distribute remaining iterations evenly
    if rem > 0
        if tid <= rem
            f = f + (tid-1)
            l = l + tid
        else
            f = f + rem
            l = l + rem
        end
    end
    # run this thread's iterations
    for i = f:l
        block = @inbounds blocks(iterspace)[i]
        ctx = mkcontext(obj, block, ndrange, iterspace, dynamic)
        Cassette.overdub(ctx, obj.f, args...)
    end
    return nothing
end

Cassette.@context CPUCtx

function mkcontext(kernel::Kernel{CPU}, I, _ndrange, iterspace, ::Dynamic) where Dynamic
    metadata = CompilerMetadata{ndrange(kernel), Dynamic}(I, _ndrange, iterspace)
    Cassette.disablehooks(CPUCtx(pass = CompilerPass, metadata=metadata))
end

@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(__index_Local_Linear), idx::CartesianIndex)
    indices = workitems(__iterspace(ctx.metadata))
    return @inbounds LinearIndices(indices)[idx]
end

@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(__index_Group_Linear), idx::CartesianIndex)
    indices = blocks(__iterspace(ctx.metadata))
    return @inbounds LinearIndices(indices)[__groupindex(ctx.metadata)]
end

@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(__index_Global_Linear), idx::CartesianIndex)
    I = @inbounds expand(__iterspace(ctx.metadata), __groupindex(ctx.metadata), idx)
    @inbounds LinearIndices(__ndrange(ctx.metadata))[I]
end

@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(__index_Local_Cartesian), idx::CartesianIndex)
    return idx
end

@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(__index_Group_Cartesian), idx::CartesianIndex)
    __groupindex(ctx.metadata)
end

@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(__index_Global_Cartesian), idx::CartesianIndex)
    return @inbounds expand(__iterspace(ctx.metadata), __groupindex(ctx.metadata), idx)
end

@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(__validindex), idx::CartesianIndex)
    # Turns this into a noop for code where we can turn of checkbounds of
    if __dynamic_checkbounds(ctx.metadata)
        I = @inbounds expand(__iterspace(ctx.metadata), __groupindex(ctx.metadata), idx)
        return I in __ndrange(ctx.metadata)
    else
        return true
    end
end


@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(__print), items...)
    __print(items...)
end

generate_overdubs(@__MODULE__, CPUCtx)

# Don't recurse into these functions
const cpufuns = (:cos, :cospi, :sin, :sinpi, :tan,
          :acos, :asin, :atan,
          :cosh, :sinh, :tanh,
          :acosh, :asinh, :atanh,
          :log, :log10, :log1p, :log2,
          :exp, :exp2, :exp10, :expm1, :ldexp,
          :isfinite, :isinf, :isnan, :signbit,
          :abs,
          :sqrt, :cbrt,
          :ceil, :floor,)
for f in cpufuns
    @eval function Cassette.overdub(ctx::CPUCtx, ::typeof(Base.$f), x::Union{Float32, Float64})
        @Base._inline_meta
        return Base.$f(x)
    end
end

@inline Cassette.overdub(::CPUCtx, ::typeof(SpecialFunctions.gamma), x::Union{Float32, Float64}) = SpecialFunctions.gamma(x)
@inline Cassette.overdub(::CPUCtx, ::typeof(SpecialFunctions.erf), x::Union{Float32, Float64}) = SpecialFunctions.erf(x)
@inline Cassette.overdub(::CPUCtx, ::typeof(SpecialFunctions.erfc), x::Union{Float32, Float64}) = SpecialFunctions.erfc(x)

###
# CPU implementation of shared memory
###
@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(SharedMemory), ::Type{T}, ::Val{Dims}, ::Val) where {T, Dims}
    MArray{__size(Dims), T}(undef)
end

###
# CPU implementation of scratch memory
# - memory allocated as a MArray with size `Dims`
###

struct ScratchArray{N, D}
    data::D
    ScratchArray{N}(data::D) where {N, D} = new{N, D}(data)
end

@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(Scratchpad), ::Type{T}, ::Val{Dims}) where {T, Dims}
    return ScratchArray{length(Dims)}(MArray{__size((Dims..., prod(__groupsize(ctx.metadata)))), T}(undef))
end

# Base.view creates a boundscheck which captures A
# https://github.com/JuliaLang/julia/issues/39308
@inline function aview(A, I::Vararg{Any, N}) where N
     J = Base.to_indices(A, I)
     Base.unsafe_view(Base._maybe_reshape_parent(A, Base.index_ndims(J...)), J...)
end

@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(Base.getindex), A::ScratchArray{N}, idx) where N
    return @inbounds aview(A.data, ntuple(_->:, Val(N))..., idx)
end

# Argument conversion
KernelAbstractions.argconvert(k::Kernel{CPU}, arg) = arg
