# The vector shuffles of `SPV_INTEL_subgroups` (`intel_sub_group_shuffle` on `uint2` to
# `uint16`), which PoCL's CPU device implements with a single work-group barrier, like a
# shuffle of a scalar. The standard shuffles only take scalars.
for N in (2, 3, 4, 8, 16)
    V = NTuple{N, VecElement{UInt32}}
    @eval @device_function @inline vector_shuffle(x::$V, lane::Int32) = @builtin_ccall(
        "__spirv_SubgroupShuffleINTEL", $V, ($V, UInt32), x, (lane - Int32(1)) % UInt32,
        convergent = true
    )
end
