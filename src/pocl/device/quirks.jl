# throw a device-side exception of type `name`, printing the type and `reason`
macro gputhrow(name::String, reason::String)
    return quote
        @println "ERROR: " $name ": " $reason "."
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
