export @groupreduce, @warpreduce

macro warpreduce(op, val)
    quote
        $__warpreduce($(esc(op)),$(esc(val)))
    end
end

macro groupreduce(op, val, neutral, conf) 
    quote
        $__groupreduce($(esc(:__ctx__)),$(esc(op)), $(esc(val)), $(esc(neutral)), typeof($(esc(val))), Val($(esc(conf)).use_warps))
    end
end

function __warpreduce(op, val)
    error("@warpreduce used outside kernel, not captured, or not supported")
end

@inline _map_getindex(args::Tuple, I) = ((args[1][I]), _map_getindex(Base.tail(args), I)...)
@inline _map_getindex(args::Tuple{Any}, I) = ((args[1][I]),)
@inline _map_getindex(args::Tuple{}, I) = ()

# groupreduction using warp intrinsics
@inline function __groupreduce(__ctx__, op, val, neutral, ::Type{T}, ::Val{true}) where {T}
    threadIdx_local = KernelAbstractions.@index(Local)
    groupsize = KernelAbstractions.@groupsize()[1]

    shared = KernelAbstractions.@localmem(T, 32)

    warpIdx, warpLane = fldmod1(threadIdx_local, 32)

    # each warp performs partial reduction
    val = KernelAbstractions.@warpreduce(op, val)

    # write reduced value to shared memory
    if warpLane == 1
        @inbounds shared[warpIdx] = val
    end

    # wait for all partial reductions
    KernelAbstractions.@synchronize()

    # read from shared memory only if that warp existed
    val = if threadIdx_local <= fld1(groupsize, 32)
            @inbounds shared[warpLane]
    else
        neutral
    end

    # final reduce within first warp
    if warpIdx == 1
        val =  KernelAbstractions.@warpreduce(op, val)
    end

    return val

end

# groupreduction using local memory
@inline function __groupreduce(__ctx__, op, val, neutral, ::Type{T}, ::Val{false}) where {T}
    threadIdx_local = KernelAbstractions.@index(Local)
    groupsize = KernelAbstractions.@groupsize()[1]
    
    shared = KernelAbstractions.@localmem(T, groupsize)

    @inbounds shared[threadIdx_local] = val

    # perform the reduction
    d = 1
    while d < groupsize
        KernelAbstractions.@synchronize()
        index = 2 * d * (threadIdx_local-1) + 1
        @inbounds if index <= groupsize
            other_val = if index + d <= groupsize
                shared[index+d]
            else
                neutral
            end
            shared[index] = op(shared[index], other_val)
        end
        d *= 2
    end

    # load the final value on the first thread
    if threadIdx_local == 1
        val = @inbounds shared[threadIdx_local]
    end
    
    return val 
end

@kernel function reduce_kernel(f, op, neutral, R, A , conf)
    # values for the kernel
    threadIdx_local = @index(Local)
    threadIdx_global = @index(Global)
    groupIdx = @index(Group)
    gridsize = @ndrange()[1]


    # load neutral value
    neutral = if neutral === nothing
        R[1]
    else
        neutral
    end
    
    val = op(neutral, neutral)

    # every thread reduces a few values parrallel
    index = threadIdx_global 
    while index <= length(A)
        val = op(val,f(A[index]))
        index += gridsize
    end
    # reduce every block to a single value
    val = @groupreduce(op, val, neutral, conf)

    # write reduces value to memory
    if threadIdx_local == 1
        R[groupIdx] = val
    end
end

function mapreducedim(f::F, op::OP, R, A::Union{AbstractArray,Broadcast.Broadcasted}; init=nothing, atomics = false, warps = false,conf=nothing) where {F, OP}

    Base.check_reducedims(R, A)
    length(A) == 0 && return R # isempty(::Broadcasted) iterates

    # add singleton dimensions to the output container, if needed
    if ndims(R) < ndims(A)
        dims = Base.fill_to_length(size(R), 1, Val(ndims(A)))
        R = reshape(R, dims)
    end
    
    backend = KernelAbstractions.get_backend(A) 
    if conf == nothing
        dev = CUDA.device()
        
        major = CUDA.capability(dev).major
        max_groupsize = if major >= 3 1024 else 512 end
        gridsize = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)        
        # supports_atomics = if warps == nothing major >= 2 else atomics end
        # supports_warp_reduce = if warps == nothing major >= 5 else warps end
        supports_atomics = atomics
        supports_warp_reduce = warps
        
        conf = Config(256, 256 * gridsize * 4, 16, supports_atomics, supports_warp_reduce)
    end

    # TODO: don't hard code these values 
    if length(R) == 1
        # only a single block needs to be launched
        if length(A) <= conf.items_per_workitem * conf.groupsize
            
            reduce_kernel(backend, conf.groupsize)(f, op, init, R, A, conf, ndrange=conf.groupsize)
            return R
        else
            ndrange = cld(length(A), conf.items_per_workitem)
            ndrange = min(gridsize, conf.max_ndrange)

            groups = cld(ndrange, conf.groupsize)
            partial = similar(R, (size(R)..., conf.use_atomics==true ? 1 : groups))

            reduce_kernel(backend, conf.groupsize)(f, op, init, partial, A, conf, ndrange=ndrange)
            
            if !conf.use_atomics
                    mapreducedim(x->x, op, R, partial;atomics=atomics, warps=warps, init=init,conf=conf)
                return R
            end

            return partial
        end
    
    else
        # # we have to use  CartesianIndices

        # # iteration domain, split in two: one part covers the dimensions that should
        # # be reduced, and the other covers the rest. combining both covers all values.
        # Rall = CartesianIndices(axes(A))
        # Rother = CartesianIndices(axes(R))
        # Rreduce = CartesianIndices(ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A)))

        # # NOTE: we hard-code `OneTo` (`first.(axes(A))` would work too) or we get a
        # #       CartesianIndices object with UnitRanges that behave badly on the GPU.
        # @assert length(Rall) == length(Rother) * length(Rreduce)
        # @assert length(Rother) > 0

        # blocks = if length(R) == 1 
        #     cld(length(A),threads)
        # else
        #     length(Rother)
        # end

        # # allocate an additional, empty dimension to write the reduced value to.
        # if length(R) == 1 
        #     partial = similar(R, (size(R)..., fld(conf.ndrange , conf.threads_per_block)))
        #     reduce_kernel(backend, group)(f, op, init, conf, Rreduce, Rother, partial, A, ndrange=conf.ndrange)

        #     print(partial)
        # else
        #     reduce_kernel(backend, conf.threads_per_block)(f, op, init, conf, Rreduce, Rother, R, A, ndrange=conf.ndrange)

        #     print(R[1])
        # end 
    end

    return R
end



