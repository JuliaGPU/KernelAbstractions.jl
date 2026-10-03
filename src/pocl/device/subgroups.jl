# Sub-group operations that SPIRVIntrinsics doesn't wrap (the way we need them).

# `sub_group_shuffle`, with the lane passed modulo `UInt32`, so that an out-of-range lane
# gives an unspecified value rather than an `InexactError`
for T in SPIRVIntrinsics.gentypes
    @eval @device_function shuffle(x::$T, lane::Integer) =
        @builtin_ccall(
        "__spirv_GroupNonUniformShuffle", $T, (UInt32, $T, UInt32),
        UInt32(Scope.Subgroup), x, (lane - 1) % UInt32
    )
end

# Votes, from `cl_khr_subgroups` and `cl_khr_subgroup_ballot`. The SPIR-V back-end lowers
# these OpenCL built-ins, which have to be listed in `subgroup_intrinsics`.
const subgroup_intrinsics = ["_Z13sub_group_anyi", "_Z13sub_group_alli", "_Z16sub_group_balloti"]

@device_function sub_group_any(pred::Bool) =
    ccall("extern _Z13sub_group_anyi", llvmcall, Int32, (Int32,), pred) != Int32(0)

@device_function sub_group_all(pred::Bool) =
    ccall("extern _Z13sub_group_alli", llvmcall, Int32, (Int32,), pred) != Int32(0)

# bit `i` of the result is set for the lane with (0-based) id `i`
@device_function sub_group_ballot(pred::Bool) =
    ccall("extern _Z16sub_group_balloti", llvmcall, NTuple{4, VecElement{UInt32}}, (Int32,), pred)
