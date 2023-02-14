import UnsafeAtomicsLLVM

function synchronize(::CPU)
    nothing
    # last = Base.get(task_local_storage(), :KA_CPU, nothing)
    # if last !== nothing
    #     wait(last)
    # end
end

@inline function do_async(f::F, args...; progress=nothing, kwargs...) where F
    # last = Base.get(task_local_storage(), :KA_CPU, nothing)
    # task = Base.Threads.@spawn begin
    #     if last !== nothing
    #         wait(last)
    #     end
    #     f(args...; kwargs...)
    #     nothing
    # end
    # task_local_storage(:KA_CPU, task)
    f(args...; kwargs...)
    return nothing
end

function async_copy!(::CPU, A, B; progress=nothing)
    do_async(copyto!, A, B; progress)
end

function (obj::Kernel{CPU})(args...; ndrange=nothing, workgroupsize=nothing, progress=nothing)
    ndrange, workgroupsize, iterspace, dynamic = launch_config(obj, ndrange, workgroupsize)

    if length(blocks(iterspace)) == 0
        return nothing
    end

    do_async(__run, obj, ndrange, iterspace, args, dynamic; progress)
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
        obj.f(ctx, args...)
    end
    return nothing
end

function mkcontext(kernel::Kernel{CPU}, I, _ndrange, iterspace, ::Dynamic) where Dynamic
    return CompilerMetadata{ndrange(kernel), Dynamic}(I, _ndrange, iterspace)
end

@inline function __index_Local_Linear(ctx, idx::CartesianIndex)
    indices = workitems(__iterspace(ctx))
    return @inbounds LinearIndices(indices)[idx]
end

@inline function __index_Group_Linear(ctx, idx::CartesianIndex)
    indices = blocks(__iterspace(ctx))
    return @inbounds LinearIndices(indices)[__groupindex(ctx)]
end

@inline function __index_Global_Linear(ctx, idx::CartesianIndex)
    I = @inbounds expand(__iterspace(ctx), __groupindex(ctx), idx)
    @inbounds LinearIndices(__ndrange(ctx))[I]
end

@inline function __index_Local_Cartesian(_, idx::CartesianIndex)
    return idx
end

@inline function __index_Group_Cartesian(ctx, ::CartesianIndex)
    __groupindex(ctx)
end

@inline function __index_Global_Cartesian(ctx, idx::CartesianIndex)
    return @inbounds expand(__iterspace(ctx), __groupindex(ctx), idx)
end

@inline function __validindex(ctx, idx::CartesianIndex)
    # Turns this into a noop for code where we can turn of checkbounds of
    if __dynamic_checkbounds(ctx)
        I = @inbounds expand(__iterspace(ctx), __groupindex(ctx), idx)
        return I in __ndrange(ctx)
    else
        return true
    end
end

###
# CPU implementation of shared memory
###
@inline function SharedMemory(::Type{T}, ::Val{Dims}, ::Val) where {T, Dims}
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

@inline function Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    return ScratchArray{length(Dims)}(MArray{__size((Dims..., prod(__groupsize(ctx)))), T}(undef))
end

# Base.view creates a boundscheck which captures A
# https://github.com/JuliaLang/julia/issues/39308
@inline function aview(A, I::Vararg{Any, N}) where N
     J = Base.to_indices(A, I)
     Base.unsafe_view(Base._maybe_reshape_parent(A, Base.index_ndims(J...)), J...)
end

@inline function Base.getindex(A::ScratchArray{N}, idx) where N
    return @inbounds aview(A.data, ntuple(_->:, Val(N))..., idx)
end

# Argument conversion
KernelAbstractions.argconvert(k::Kernel{CPU}, arg) = arg
