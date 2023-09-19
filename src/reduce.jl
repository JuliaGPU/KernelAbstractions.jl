export @groupreduce

"""
    @groupreduce(op, val, neutral, use_subgroups)

Reduce values across a block
- `op`: the operator of the reduction
- `val`: value that each thread contibutes to the values that need to be reduced
- `neutral`: value of the operator, so that `op(neutral, neutral) = neutral`
- `use_subgroups`: make use of the subgroupreduction of the groupreduction
"""
macro groupreduce(op, val, neutral) 
    quote
        $__groupreduce($(esc(:__ctx__)),$(esc(op)), $(esc(val)), $(esc(neutral)), $(esc(typeof(val))))
    end
end

@inline function __groupreduce(__ctx__, op, val, neutral, ::Type{T}) where {T}
    idx_in_group = @index(Local, Linear)
    groupsize = prod(@groupsize())
    
    localmem = @localmem(T, groupsize)

    @inbounds localmem[idx_in_group] = val

    # perform the reduction
    d = 1
    while d < groupsize
        @synchronize()
        index = 2 * d * (idx_in_group-1) + 1
        @inbounds if index <= groupsize
            other_val = if index + d <= groupsize
                localmem[index+d]
            else
                neutral
            end
            localmem[index] = op(localmem[index], other_val)
        end
        d *= 2
    end

    # load the final value on the first thread
    if idx_in_group == 1
        val = @inbounds localmem[idx_in_group]
    end
    
    return val 
end
