struct CPUEvent <: Event
    task::Core.Task
end

function wait(ev::CPUEvent, progress=nothing)
    if progress === nothing
        wait(ev.task)
    else
        while !Base.istaskdone(ev.task)
            progress()
            yield() # yield to the scheduler
        end
    end
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

    t = __run(obj, ndrange, iterspace, args, dependencies)
    return CPUEvent(t)
end

# Inference barrier
function __run(obj, ndrange, iterspace, args, dependencies)
    return Threads.@spawn begin
        if dependencies !== nothing
            cpu_tasks = Core.Task[]
            for event in dependencies
                if event isa CPUEvent
                    push!(cpu_tasks, event.task)
                end
            end
            !isempty(cpu_tasks) && Base.sync_end(cpu_tasks)
            for event in dependencies
                if !(event isa CPUEvent)
                    wait(event, ()->yield())
                end
            end
        end
        @sync begin
            # TODO: how do we use the information that the iteration space maps perfectly to
            #       the ndrange without incurring a 2x compilation overhead
            # if dynamic
                for block in iterspace
                    let ctx = mkcontextdynamic(obj, block, ndrange, iterspace)
                        Threads.@spawn Cassette.overdub(ctx, obj.f, args...)
                    end
                end
            # else
            #     for block in iterspace
            #         let ctx = mkcontext(obj, blocks, ndrange, iterspace)
            #             Threads.@spawn Cassette.overdub(ctx, obj.f, args...)
            #         end
            #     end
            # end
        end
    end
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
