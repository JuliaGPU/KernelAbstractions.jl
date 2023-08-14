export groupreduce, warpreduce

macro groupreduce(op, val, neutral, conf) 
    quote
        $__groupreduce($(esc(:__ctx__)),$(esc(op)), $(esc(val)), $(esc(neutral)), typeof($(esc(val))), Val($(esc(conf)).use_warps))
    end
end

macro warpreduce(op, val)
    quote
        $__warpreduce(esc(op))($(esc(val)))
    end
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

@kernel function reduce_kernel(f, op, neutral, grain, R, A , conf)
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
        val = op(val,A[index])
        index += gridsize
    end

    # reduce every block to a single value
    val = @reduce(op, val, neutral, conf)

    # write reduces value to memory
    if threadIdx_local == 1
        R[groupIdx] = val
    end
end
