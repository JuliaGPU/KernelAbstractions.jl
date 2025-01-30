"""
    @groupreduce algo op val neutral [groupsize]

Perform group reduction of `val` using `op`.

# Arguments

- `algo` specifies which reduction algorithm to use:
    - `:thread`:
        Perform thread group reduction (requires `groupsize * sizeof(T)` bytes of shared memory).
        Available accross all backends.
    - `:warp`:
        Perform warp group reduction (requires `32 * sizeof(T)` bytes of shared memory).

- `neutral` should be a neutral w.r.t. `op`, such that `op(neutral, x) == x`.
- `groupsize` specifies size of the workgroup.
    If a kernel does not specifies `groupsize` statically, then it is required to
    provide `groupsize`.
    Also can be used to perform reduction accross first `groupsize` threads
    (if `groupsize < @groupsize()`).

# Returns

Result of the reduction.
"""
macro groupreduce(algo, op, val, neutral)
    f = if algo.value == :thread
        __groupreduce
    elseif algo.value == :warp
        __warp_groupreduce
    else
        error(
            "@groupreduce supports only :thread or :warp as a reduction algorithm, " *
            "but $(algo.value) was specified.")
    end
    quote
        $f(
            $(esc(:__ctx__)),
            $(esc(op)),
            $(esc(val)),
            $(esc(neutral)),
            Val(prod($groupsize($(esc(:__ctx__))))),
        )
    end
end

macro groupreduce(algo, op, val, neutral, groupsize)
    f = if algo.value == :thread
        __groupreduce
    elseif algo.value == :warp
        __warp_groupreduce
    else
        error(
            "@groupreduce supports only :thread or :warp as a reduction algorithm, " *
            "but $(algo.value) was specified.")
    end
    quote
        $f(
            $(esc(:__ctx__)),
            $(esc(op)),
            $(esc(val)),
            $(esc(neutral)),
            Val($(esc(groupsize))),
        )
    end
end

function __groupreduce(__ctx__, op, val::T, neutral::T, ::Val{groupsize}) where {T, groupsize}
    storage = @localmem T groupsize

    local_idx = @index(Local)
    local_idx ≤ groupsize && (storage[local_idx] = val)
    @synchronize()

    s::UInt64 = groupsize ÷ 0x2
    while s > 0x0
        if (local_idx - 0x1) < s
            other_idx = local_idx + s
            if other_idx ≤ groupsize
                storage[local_idx] = op(storage[local_idx], storage[other_idx])
            end
        end
        @synchronize()
        s >>= 0x1
    end

    if local_idx == 0x1
        val = storage[local_idx]
    end
    return val
end

# Warp groupreduce.

macro shfl_down(val, offset)
    quote
        $__shfl_down($(esc(val)), $(esc(offset)))
    end
end

# Backends should implement this.
function __shfl_down end

@inline function __warp_reduce(val, op)
    offset::UInt32 = UInt32(32) ÷ 0x2
    while offset > 0x0
        val = op(val, @shfl_down(val, offset))
        offset >>= 0x1
    end
    return val
end

# Assume warp is 32 lanes.
const __warpsize::UInt32 = 32
# Maximum number of warps (for a groupsize = 1024).
const __warp_bins::UInt32 = 32

function __warp_groupreduce(__ctx__, op, val::T, neutral::T, ::Val{groupsize}) where {T, groupsize}
    storage = @localmem T __warp_bins

    local_idx = @index(Local)
    lane = (local_idx - 0x1) % __warpsize + 0x1
    warp_id = (local_idx - 0x1) ÷ __warpsize + 0x1

    # Each warp performs a reduction and writes results into its own bin in `storage`.
    val = __warp_reduce(val, op)
    lane == 0x1 && (storage[warp_id] = val)
    @synchronize()

    # Final reduction of the `storage` on the first warp.
    within_storage = (local_idx - 0x1) < groupsize ÷ __warpsize
    val = within_storage ? storage[lane] : neutral
    warp_id == 0x1 && (val = __warp_reduce(val, op))
    return val
end
