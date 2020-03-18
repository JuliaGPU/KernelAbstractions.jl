struct CPUEvent <: Event
    task::Core.Task
    bt
    CPUEvent(ev) = new(ev, Base.backtrace())
end

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

wait(ev::Union{CPUEvent, NoneEvent, MultiEvent}, progress=yield) = wait(CPU(), ev, progress)
wait(::CPU, ev::NoneEvent, progress=yield) = (progress(); nothing)

function wait(cpu::CPU, ev::MultiEvent, progress=yield)
    @info "Waiting on MultiEvent" origin=stacktrace(ev.bt) current=stacktrace()

    dependencies = collect(ev.events)
    cpudeps   = filter(d->d isa CPUEvent && d.task !== nothing, dependencies)
    otherdeps = filter(d->!(d isa CPUEvent), dependencies)
    Base.sync_end(map(e->e.task, cpudeps))
    for event in otherdeps
        wait(CPU(), event, progress)
    end
end

function wait(::CPU, ev::CPUEvent, progress=yield)
    token = gensym()
    @info "Waiting on CPUEvent" origin=stacktrace(ev.bt) current=stacktrace() token

    if progress === nothing
        wait(ev.task)
    else
        t = ev.task
        if !Base.istaskdone(t)
            lock(t.donenotify)
            try
                while !Base.istaskdone(t)
                    @info "Still waiting on CPUEvent" token
                    progress()
                    wait(t.donenotify)
                end
            finally
                unlock(t.donenotify)
            end
        end
    end
end

function async_copy!(::CPU, A, B; dependencies=nothing, progress=yield)
    Event(copyto!, A, B, dependencies=dependencies, progress=progress)
end

function (obj::Kernel{CPU})(args...; ndrange=nothing, workgroupsize=nothing, dependencies=nothing, progress=yield)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize, )
    end
    if dependencies isa Event
        dependencies = (dependencies,)
    end

    if KernelAbstractions.workgroupsize(obj) <: DynamicSize && workgroupsize === nothing
        workgroupsize = (1024,) # Vectorization, 4x unrolling, minimal grain size
    end
    iterspace, dynamic = partition(obj, ndrange, workgroupsize)
    # partition checked that the ndrange's agreed
    if KernelAbstractions.ndrange(obj) <: StaticSize
        ndrange = nothing
    end

    Event(__run, obj, ndrange, iterspace, args, Val(dynamic),
          dependencies=dependencies, progress=progress)
end

# Inference barriers
function __run(obj, ndrange, iterspace, args, ::Val{dynamic}) where dynamic
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
        __thread_run(1, len, rem, obj, ndrange, iterspace, args, Val(dynamic))
    else
        @sync for tid in 1:Nthreads
            Threads.@spawn __thread_run(tid, len, rem, obj, ndrange, iterspace, args, Val(dynamic))
        end
    end
    return nothing
end

function __thread_run(tid, len, rem, obj, ndrange, iterspace, args, ::Val{dynamic}) where dynamic
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
        ctx = mkcontext(obj, block, ndrange, iterspace, Val(dynamic))
        Cassette.overdub(ctx, obj.f, args...)
    end
    return nothing
end

Cassette.@context CPUCtx

function mkcontext(kernel::Kernel{CPU}, I, _ndrange, iterspace, ::Val{dynamic}) where dynamic
    metadata = CompilerMetadata{ndrange(kernel), dynamic}(I, _ndrange, iterspace)
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

generate_overdubs(CPUCtx)

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


###
# CPU implementation of shared memory
###
@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(SharedMemory), ::Type{T}, ::Val{Dims}, ::Val) where {T, Dims}
    MArray{__size(Dims), T}(undef)
end

###
# CPU implementation of scratch memory
# - private memory for each workitem
# - memory allocated as a MArray with size `Dims + WorkgroupSize`
###
struct ScratchArray{N, D}
    data::D
    ScratchArray{N}(data::D) where {N, D} = new{N, D}(data)
end

@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(Scratchpad), ::Type{T}, ::Val{Dims}) where {T, Dims}
    return ScratchArray{length(Dims)}(MArray{__size((Dims..., __groupsize(ctx.metadata)...)), T}(undef))
end

Base.@propagate_inbounds function Base.getindex(A::ScratchArray, I...)
    return A.data[I...]
end

Base.@propagate_inbounds function Base.setindex!(A::ScratchArray, val, I...)
    A.data[I...] = val
end
