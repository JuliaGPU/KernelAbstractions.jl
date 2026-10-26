# throw a device-side exception of type `name`, reporting it with `reason`
macro gputhrow(name::String, reason::String)
    escape(str) = replace(str, "%" => "%%")
    fmt = "ERROR: $(escape(name)) during kernel execution on work-item (%ld, %ld, %ld): $(escape(reason)).\n"
    return quote
        if GPUCompiler.kernel_debug_level() >= 1 && claim_output() == 1
            SPIRVIntrinsics.@printf($fmt, get_global_id(1), get_global_id(2), get_global_id(3))
        end
        throw(nothing)
    end
end

# math.jl
@device_override @noinline Base.Math.throw_complex_domainerror(f::Symbol, x) =
    @gputhrow "DomainError" "This operation requires a complex input to return a complex result"
@device_override @noinline Base.Math.throw_exp_domainerror(x) =
    @gputhrow "DomainError" "Exponentiation yielding a complex result requires a complex argument"

# intfuncs.jl
@device_override @noinline Base.throw_domerr_powbysq(::Any, p) =
    @gputhrow "DomainError" "Cannot raise an integer to a negative power"
@device_override @noinline Base.throw_domerr_powbysq(::Integer, p) =
    @gputhrow "DomainError" "Cannot raise an integer to a negative power"
@device_override @noinline Base.throw_domerr_powbysq(::AbstractMatrix, p) =
    @gputhrow "DomainError" "Cannot raise an integer to a negative power"

# checked.jl
@device_override @noinline Base.Checked.throw_overflowerr_binaryop(op, x, y) =
    @gputhrow "OverflowError" "Binary operation overflowed"

# boot.jl
@device_override @noinline Core.throw_inexacterror(f::Symbol, ::Type{T}, val) where {T} =
    @gputhrow "InexactError" "Inexact conversion"

# abstractarray.jl
@device_override @noinline Base.throw_boundserror(A, I) =
    @gputhrow "BoundsError" "Out-of-bounds array access"

# essentials.jl
# Julia 1.14 routes indexed bounds errors through `_throw_boundserror_indices`
# rather than `throw_boundserror`, bypassing the override above.
@static if isdefined(Base, :_throw_boundserror_indices)
    @device_override @noinline Base._throw_boundserror_indices(A) =
        @gputhrow "BoundsError" "Out-of-bounds array access"
    @device_override @noinline Base._throw_boundserror_indices(A, i1, I...) =
        @gputhrow "BoundsError" "Out-of-bounds array access"
end

# range.jl
# From Metal.jl to avoid widemul and Int128, which the SPIR-V back-end cannot lower.
# Unlike Metal.jl, this covers all the types Base's method does: `widemul` of a range of
# `Int32` and an `Int64` index widens to `Int128` as well.
@static if VERSION >= v"1.12.0-DEV.1736" # Partially reverts JuliaLang/julia PR #56750
    const BitInteger64 = Union{Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64}
    @device_override function Base.checkbounds(::Type{Bool}, v::StepRange{<:BitInteger64, <:BitInteger64}, i::BitInteger64)
        @inline
        return checkindex(Bool, eachindex(IndexLinear(), v), i)
    end
end

# trig.jl
@device_override @noinline Base.Math.sincos_domain_error(x) =
    @gputhrow "DomainError" "sincos(x) is only defined for finite x"

# diagonal.jl
# XXX: remove when we have malloc
# import LinearAlgebra
# @device_override function Base.setindex!(D::LinearAlgebra.Diagonal, v, i::Int, j::Int)
#     @boundscheck checkbounds(D, i, j)
#     if i == j
#         @inbounds D.diag[i] = v
#     elseif !iszero(v)
#         @gputhrow "ArgumentError" "cannot set off-diagonal entry to a nonzero value"
#     end
#     return v
# end

# number.jl
# XXX: remove when we have malloc
@device_override @inline function Base.getindex(x::Number, I::Integer...)
    @boundscheck all(isone, I) ||
        @gputhrow "BoundsError" "Out-of-bounds access of scalar value"
    x
end

# float.jl
# `muladd` allows contracting the multiplication and addition, but SPIR-V can only express
# that with `SPV_KHR_float_controls2`, which we don't emit and the PoCL JLL can't read, so
# PoCL computes it as a separate multiplication and addition. on CPUs with hardware FMA,
# fuse them explicitly. these overlays are only used for jobs that set `fuse_muladd` (see
# `fast_fma`). they are not consistent with Base's `muladd`, which may or may not fuse.
# XXX: a stopgap (#865). remove it, and `fuse_muladd`, once the PoCL JLL is built with
#      LLVM 23, whose SPIR-V translator reads the extension (pocl/pocl#2393), and we enable
#      the extension for devices that report it (JuliaGPU/GPUCompiler.jl#1007).
Base.Experimental.@MethodTable(muladd_method_table)
for T in (Float32, Float64)
    @eval Base.Experimental.@overlay muladd_method_table Base.muladd(x::$T, y::$T, z::$T) =
        Base.fma(x, y, z)
end
