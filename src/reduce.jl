export @groupreduce, Reduction

module Reduction
    const thread = Val(:thread)
    const warp = Val(:warp)
end

"""
    @groupreduce op val neutral algo [groupsize]

Perform group reduction of `val` using `op`.

# Arguments

- `algo` specifies which reduction algorithm to use:
    - `Reduction.thread`:
        Perform thread group reduction (requires `groupsize * sizeof(T)` bytes of shared memory).
        Available accross all backends.
    - `Reduction.warp`:
        Perform warp group reduction (requires `32 * sizeof(T)` bytes of shared memory).
        Potentially faster, since requires fewer writes to shared memory.
        To query if backend supports warp reduction, use `supports_warp_reduction(backend)`.

- `neutral` should be a neutral w.r.t. `op`, such that `op(neutral, x) == x`.

- `groupsize` specifies size of the workgroup.
    If a kernel does not specifies `groupsize` statically, then it is required to
    provide `groupsize`.
    Also can be used to perform reduction accross first `groupsize` threads
    (if `groupsize < @groupsize()`).

# Returns

Result of the reduction.
"""
macro groupreduce(op, val, neutral, algo)
    return quote
        __groupreduce(
            $(esc(:__ctx__)),
            $(esc(op)),
            $(esc(val)),
            $(esc(neutral)),
            Val(prod($groupsize($(esc(:__ctx__))))),
            $(esc(algo)),
        )
    end
end

macro groupreduce(op, val, neutral, algo, groupsize)
    return quote
        __groupreduce(
            $(esc(:__ctx__)),
            $(esc(op)),
            $(esc(val)),
            $(esc(neutral)),
            Val($(esc(groupsize))),
            $(esc(algo)),
        )
    end
end

function __groupreduce(__ctx__, op, val::T, neutral::T, ::Val{groupsize}, ::Val{:thread}) where {T, groupsize}
    storage = @localmem T groupsize

    local_idx = @index(Local)
    @inbounds local_idx ≤ groupsize && (storage[local_idx] = val)
    @synchronize()

    s::UInt64 = groupsize ÷ 0x02
    while s > 0x00
        if (local_idx - 0x01) < s
            other_idx = local_idx + s
            if other_idx ≤ groupsize
                @inbounds storage[local_idx] = op(storage[local_idx], storage[other_idx])
            end
        end
        @synchronize()
        s >>= 0x01
    end

    if local_idx == 0x01
        @inbounds val = storage[local_idx]
    end
    return val
end

# Warp groupreduce.

macro shfl_down(val, offset)
    return quote
        $__shfl_down($(esc(val)), $(esc(offset)))
    end
end

# Backends should implement these two.
function __shfl_down end
supports_warp_reduction(::CPU) = false

@inline function __warp_reduce(val, op)
    offset::UInt32 = UInt32(32) ÷ 0x02
    while offset > 0x00
        val = op(val, @shfl_down(val, offset))
        offset >>= 0x01
    end
    return val
end

# Assume warp is 32 lanes.
const __warpsize::UInt32 = 32
# Maximum number of warps (for a groupsize = 1024).
const __warp_bins::UInt32 = 32

function __groupreduce(__ctx__, op, val::T, neutral::T, ::Val{groupsize}, ::Val{:warp}) where {T, groupsize}
    storage = @localmem T __warp_bins

    local_idx = @index(Local)
    lane = (local_idx - 0x01) % __warpsize + 0x01
    warp_id = (local_idx - 0x01) ÷ __warpsize + 0x01

    # Each warp performs a reduction and writes results into its own bin in `storage`.
    val = __warp_reduce(val, op)
    @inbounds lane == 0x01 && (storage[warp_id] = val)
    @synchronize()

    # Final reduction of the `storage` on the first warp.
    within_storage = (local_idx - 0x01) < groupsize ÷ __warpsize
    @inbounds val = within_storage ? storage[lane] : neutral
    warp_id == 0x01 && (val = __warp_reduce(val, op))
    return val
end
