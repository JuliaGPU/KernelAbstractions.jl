# Replacements for Base methods that throw, so that the device records what went wrong
# (see `@gputhrow` in `device/runtime.jl`) instead of only signalling that it failed.

# math.jl
@device_override @noinline Base.Math.throw_complex_domainerror(f::Symbol, x) =
    @gputhrow "DomainError" "This operation requires a complex input to return a complex result"
@device_override @noinline Base.Math.throw_exp_domainerror(x) =
    @gputhrow "DomainError" "Exponentiation yielding a complex result requires a complex argument"
@static if isdefined(Base.Math, :throw_finite_domainerror)
    @device_override @noinline Base.Math.throw_finite_domainerror(f::Symbol, x) =
        @gputhrow "DomainError" "function is only defined for finite x."
end
@device_override function Base.Math.exponent(x::T) where {T <: Base.IEEEFloat}
    xs = reinterpret(Unsigned, x) & ~Base.sign_mask(T)
    xs >= Base.exponent_mask(T) && @gputhrow "DomainError" "Cannot be NaN or Inf."
    k = Int(xs >> Base.significand_bits(T))
    if k == 0 # x is subnormal
        xs == 0 && @gputhrow "DomainError" "Cannot be ±0.0."
        m = leading_zeros(xs) - Base.exponent_bits(T)
        k = 1 - m
    end
    return k - Base.exponent_bias(T)
end

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
@device_override @noinline Base.Checked.throw_overflowerr_negation(op, x, y) =
    @gputhrow "OverflowError" "Negation overflowed"

# boot.jl
@device_override @noinline Core.throw_inexacterror(f::Symbol, ::Type{T}, val) where {T} =
    @gputhrow "InexactError" "Inexact conversion"

# bool.jl / float.jl
# `Bool(::Real)` and `Bool(::Float16)` don't go through `throw_inexacterror` but construct
# the `InexactError` themselves, through its vararg `@nospecialize` constructor. On the
# device that means boxing the arguments on the throwing branch (an allocation the
# optimizer can't remove), so route them through `@gputhrow` instead.
for T in (Real, Float16)
    @eval @device_override function Base.Bool(x::$T)
        x == 0 && return false
        x == 1 && return true
        @gputhrow "InexactError" "Inexact conversion"
    end
end

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

# trig.jl
@device_override @noinline Base.Math.sincos_domain_error(x) =
    @gputhrow "DomainError" "sincos(x) is only defined for finite x."

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
    return x
end
