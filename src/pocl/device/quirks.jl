macro print_and_throw(args...)
    return quote
        @println "ERROR: " $(args...) "."
        throw(nothing)
    end
end

# math.jl
@device_override @noinline Base.Math.throw_complex_domainerror(f::Symbol, x) =
    @print_and_throw "This operation requires a complex input to return a complex result"
@device_override @noinline Base.Math.throw_exp_domainerror(x) =
    @print_and_throw "Exponentiation yielding a complex result requires a complex argument"

# intfuncs.jl
@device_override @noinline Base.throw_domerr_powbysq(::Any, p) =
    @print_and_throw "Cannot raise an integer to a negative power"
@device_override @noinline Base.throw_domerr_powbysq(::Integer, p) =
    @print_and_throw "Cannot raise an integer to a negative power"
@device_override @noinline Base.throw_domerr_powbysq(::AbstractMatrix, p) =
    @print_and_throw "Cannot raise an integer to a negative power"

# checked.jl
@device_override @noinline Base.Checked.throw_overflowerr_binaryop(op, x, y) =
    @print_and_throw "Binary operation overflowed"
@device_override @noinline Base.Checked.throw_overflowerr_negation(op, x, y) =
    @print_and_throw "Negation overflowed"

# boot.jl
@device_override @noinline Core.throw_inexacterror(f::Symbol, ::Type{T}, val) where {T} =
    @print_and_throw "Inexact conversion"

# abstractarray.jl
@device_override @noinline Base.throw_boundserror(A, I) =
    @print_and_throw "Out-of-bounds array access"

# trig.jl
@device_override @noinline Base.Math.sincos_domain_error(x) =
    @print_and_throw "sincos(x) is only defined for finite x."

# diagonal.jl
# XXX: remove when we have malloc
# import LinearAlgebra
# @device_override function Base.setindex!(D::LinearAlgebra.Diagonal, v, i::Int, j::Int)
#     @boundscheck checkbounds(D, i, j)
#     if i == j
#         @inbounds D.diag[i] = v
#     elseif !iszero(v)
#         @print_and_throw "cannot set off-diagonal entry to a nonzero value"
#     end
#     return v
# end

# number.jl
# XXX: remove when we have malloc
@device_override @inline function Base.getindex(x::Number, I::Integer...)
    @boundscheck all(isone, I) ||
        @print_and_throw "Out-of-bounds access of scalar value"
    x
end
