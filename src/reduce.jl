export @groupreduce, @subgroupreduce

"""

@subgroupreduce(op, val)

reduce values across a subgroup. This operation is only supported if subgroups are supported by the backend.
"""
macro subgroupreduce(op, val)
    quote
        $__subgroupreduce($(esc(op)),$(esc(val)))
    end
end

function __subgroupreduce(op, val)
    error("@subgroupreduce used outside kernel, not captured, or not supported")
end

"""

@groupreduce(op, val, neutral, use_subgroups)

Reduce values across a block
- `op`: the operator of the reduction
- `val`: value that each thread contibutes to the values that need to be reduced
- `netral`: value of the operator, so that `op(netural, neutral) = neutral``
- `use_subgroups`: make use of the subgroupreduction of the groupreduction
"""
macro groupreduce(op, val, neutral, use_subgroups) 
    quote
        $__groupreduce($(esc(:__ctx__)),$(esc(op)), $(esc(val)), $(esc(neutral)), $(esc(typeof(val))), Val(use_subgroups))
    end
end

@inline function __groupreduce(__ctx__, op, val, neutral, ::Type{T}, ::Val{true}) where {T}
    idx_in_group = @index(Local)
    groupsize = @groupsize()[1]
    subgroupsize = @subgroupsize()

    localmem = @localmem(T, subgroupsize)

    idx_subgroup, idx_in_subgroup = fldmod1(idx_in_group, subgroupsize)

    # first subgroup reduction
    val = @subgroupreduce(op, val)

    # store partial results in local memory
    if idx_in_subgroup == 1
        @inbounds localmem[idx_in_subgroup] = val
    end

    @synchronize()

    val = if idx_in_subgroup <= fld1(groupsize, subgroupsize)
            @inbounds localmem[idx_in_subgroup]
    else
        neutral
    end

    # second subgroup reduction to reduce partial results
    if idx_in_subgroup == 1
        val =  @subgroupreduce(op, val)
    end

    return val
end

@inline function __groupreduce(__ctx__, op, val, neutral, ::Type{T}, ::Val{false}) where {T}
    idx_in_group = @index(Local)
    groupsize = @groupsize()[1]
    
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