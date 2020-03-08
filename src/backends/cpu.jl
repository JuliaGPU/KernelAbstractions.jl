struct CPUEvent <: Event
    task::Union{Nothing, Core.Task}
end

function Event(::CPU)
    return CPUEvent(nothing)
end

wait(ev::CPUEvent, progress=nothing) = wait(CPU(), ev, progress)
function wait(::CPU, ev::CPUEvent, progress=nothing)
    ev.task === nothing && return
    
    if progress === nothing
        wait(ev.task)
    else
        while !Base.istaskdone(ev.task)
            progress()
        end
    end
end
function __waitall(::CPU, dependencies, progress)
    if dependencies isa Event
        dependencies = (dependencies,)
    end
    if dependencies !== nothing
        dependencies = collect(dependencies)
        cpudeps   = filter(d->d isa CPUEvent && d.task !== nothing, dependencies)
        otherdeps = filter(d->!(d isa CPUEvent), dependencies)
        Base.sync_end(map(e->e.task, cpudeps))
        for event in otherdeps
            wait(CPU(), event, progress)
        end
    end
end

function async_copy!(::CPU, A, B; dependencies=nothing)
    __waitall(CPU(), dependencies, yield)
    copyto!(A, B)
    return CPUEvent(nothing)
end

function (obj::Kernel{CPU})(args...; ndrange=nothing, workgroupsize=nothing, dependencies=nothing)
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

    t = Threads.@spawn __run(obj, ndrange, iterspace, args, dependencies)
    return CPUEvent(t)
end

# Inference barriers
function __run(obj, ndrange, iterspace, args, dependencies)
    __waitall(CPU(), dependencies, yield)
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
        __thread_run(1, len, rem, obj, ndrange, iterspace, args)
    else
        @sync for tid in 1:Nthreads
            Threads.@spawn __thread_run(tid, len, rem, obj, ndrange, iterspace, args)
        end
    end
    return nothing
end

function __thread_run(tid, len, rem, obj, ndrange, iterspace, args)
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
        # TODO: how do we use the information that the iteration space maps perfectly to
        #       the ndrange without incurring a 2x compilation overhead
        # if dynamic
        ctx = mkcontextdynamic(obj, block, ndrange, iterspace)
        Cassette.overdub(ctx, obj.f, args...)
        # else
        #     ctx = mkcontext(obj, blocks, ndrange, iterspace)
        #     Threads.@spawn Cassette.overdub(ctx, obj.f, args...)
    end
    return nothing
end

Cassette.@context CPUCtx

function mkcontext(kernel::Kernel{CPU}, I, _ndrange, iterspace)
    metadata = CompilerMetadata{ndrange(kernel), false}(I, _ndrange, iterspace)
    Cassette.disablehooks(CPUCtx(pass = CompilerPass, metadata=metadata))
end

function mkcontextdynamic(kernel::Kernel{CPU}, I, _ndrange, iterspace)
    metadata = CompilerMetadata{ndrange(kernel), true}(I, _ndrange, iterspace)
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

generate_overdubs(CPUCtx)

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
