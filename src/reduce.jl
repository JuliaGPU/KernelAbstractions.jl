struct Config{
    THREADS_PER_WARP,         # size of warp 
    THREADS_PER_BLOCK         # size of blocks
   }
end

@inline function Base.getproperty(conf::Type{Config{ THREADS_PER_WARP, THREADS_PER_BLOCK}}, sym::Symbol) where { THREADS_PER_WARP, THREADS_PER_BLOCK}
    if sym == :threads_per_warp
        THREADS_PER_WARP
    elseif sym == :threads_per_block
        THREADS_PER_BLOCK
    else
        # fallback for nothing
        getfield(conf, sym)
    end
end

# TODO: make variable block size possible
# TODO: figure out where to place this
# reduction functionality for a group
@inline function __reduce(__ctx__ , op, val, neutral, ::Type{T}) where {T}
    threads = KernelAbstractions.@groupsize()[1]
    threadIdx = KernelAbstractions.@index(Local)

    # shared mem for a complete reduction
    shared = KernelAbstractions.@localmem(T, 1024)
    @inbounds shared[threadIdx] = val

    # perform the reduction
    d = 1
    while d < threads
        KernelAbstractions.@synchronize()
        index = 2 * d * (threadIdx-1) + 1
        @inbounds if index <= threads
            other_val = if index + d <= threads
                shared[index+d]
            else
                neutral
            end
            shared[index] = op(shared[index], other_val)
        end
        d *= 2
    end

    # load the final value on the first thread
    if threadIdx == 1
        val = @inbounds shared[threadIdx]
    end

    # every thread will return the reduced value of the group
    return val
end
