struct CPUEvent <: Event
    task::Core.Task
end

function wait(ev::CPUEvent)
    wait(ev.task)
end

function (obj::Kernel{CPU})(args...; ndrange=nothing, workgroupsize=nothing, dependencies=nothing)
    if ndrange isa Int
        ndrange = (ndrange,)
    end
    if dependencies isa Event
        dependencies = (dependencies,)
    end
    if KernelAbstractions.workgroupsize(obj) <: DynamicSize && workgroupsize === nothing
        workgroupsize = 1024 # Vectorization, 4x unrolling, minimal grain size
    end
    nblocks, dynamic = partition(obj, ndrange, workgroupsize)
    # partition checked that the ndrange's agreed
    if KernelAbstractions.ndrange(obj) <: StaticSize
        ndrange = nothing
    end
    if KernelAbstractions.workgroupsize(obj) <: StaticSize
        workgroupsize = nothing
    end
    t = Threads.@spawn begin
        if dependencies !== nothing
            Base.sync_end(map(e->e.task, dependencies))
        end
        @sync begin
            for I in 1:(nblocks-1)
                let ctx = mkcontext(obj, I, ndrange, workgroupsize)
                    Threads.@spawn Cassette.overdub(ctx, obj.f, args...)
                end
            end

            if dynamic
                let ctx = mkcontextdynamic(obj, nblocks, ndrange, workgroupsize)
                    Threads.@spawn Cassette.overdub(ctx, obj.f, args...)
                end
            else 
                let ctx = mkcontext(obj, nblocks, ndrange, workgroupsize)
                    Threads.@spawn Cassette.overdub(ctx, obj.f, args...)
                end
            end
        end
    end
    return CPUEvent(t)
end

Cassette.@context CPUCtx

function mkcontext(kernel::Kernel{CPU}, I, _ndrange, _workgroupsize)
    metadata = CompilerMetadata{workgroupsize(kernel), ndrange(kernel), false}(I, _ndrange, workgroupsize)
    Cassette.disablehooks(CPUCtx(pass = CompilerPass, metadata=metadata))
end

function mkcontextdynamic(kernel::Kernel{CPU}, I, _ndrange, _workgroupsize)
    metadata = CompilerMetadata{workgroupsize(kernel), ndrange(kernel), true}(I, _ndrange, workgroupsize)
    Cassette.disablehooks(CPUCtx(pass = CompilerPass, metadata=metadata))
end

@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(__index_Local_Linear), idx)
    return idx
end

@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(__index_Global_Linear), idx)
    workgroup = __groupindex(ctx.metadata)
    (workgroup - 1) * __groupsize(ctx.metadata) + idx
end

@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(__index_Local_Cartesian), idx)
    error("@index(Local, Cartesian) is not yet defined")
end

@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(__index_Global_Cartesian), idx)
    workgroup = __groupindex(ctx.metadata)
    indices = __ndrange(ctx.metadata)
    lI = (workgroup - 1) * __groupsize(ctx.metadata) + idx
    return @inbounds indices[lI]
end

@inline function Cassette.overdub(ctx::CPUCtx, ::typeof(__validindex), idx)
    # Turns this into a noop for code where we can turn of checkbounds of
    if __dynamic_checkbounds(ctx.metadata)
        maxidx = prod(size(__ndrange(ctx.metadata)))
        valid  = idx <= mod1(maxidx, __groupsize(ctx.metadata)) 
        return valid
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
    return ScratchArray{length(Dims)}(MArray{__size((Dims..., __groupsize(ctx.metadata))), T}(undef))
end

Base.@propagate_inbounds function Cassette.overdub(ctx::CPUCtx, ::typeof(Base.getindex), A::ScratchArray{N}, I...) where N
    nI = ntuple(Val(N+1)) do i
        if i == N+1
            __groupindex(ctx.metadata)
        else
            I[i]
        end
    end

    return A.data[nI...]
end

Base.@propagate_inbounds function Cassette.overdub(ctx::CPUCtx, ::typeof(Base.setindex!), A::ScratchArray{N}, val, I...) where N
    nI = ntuple(Val(N+1)) do i
        if i == N+1
            __groupindex(ctx.metadata)
        else
            I[i]
        end
    end
    A.data[nI...] = val
end
