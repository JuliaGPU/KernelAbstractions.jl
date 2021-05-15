struct CompilerMetadata{StaticNDRange, CheckBounds, I, NDRange, Iterspace}
    groupindex::I
    ndrange::NDRange
    iterspace::Iterspace

    # CPU variant
    function CompilerMetadata{NDRange, CB}(idx, ndrange, iterspace) where {NDRange, CB}
        if ndrange !== nothing
            ndrange = CartesianIndices(ndrange)
        end
        return new{NDRange, CB, typeof(idx), typeof(ndrange), typeof(iterspace)}(idx, ndrange, iterspace)
    end

    # GPU variante: index is given implicit
    function CompilerMetadata{NDRange, CB}(ndrange, iterspace) where {NDRange, CB}
        if ndrange !== nothing
            ndrange = CartesianIndices(ndrange)
        end
        return new{NDRange, CB, Nothing, typeof(ndrange), typeof(iterspace)}(nothing, ndrange, iterspace)
    end
end

@inline __iterspace(cm::CompilerMetadata)  = cm.iterspace
@inline __groupindex(cm::CompilerMetadata) = cm.groupindex
@inline __groupsize(cm::CompilerMetadata) = size(workitems(__iterspace(cm)))
@inline __dynamic_checkbounds(::CompilerMetadata{NDRange, CB}) where {NDRange, CB} = CB <: DynamicCheck
@inline __ndrange(cm::CompilerMetadata{NDRange}) where {NDRange<:StaticSize}  = CartesianIndices(get(NDRange))
@inline __ndrange(cm::CompilerMetadata{NDRange}) where {NDRange<:DynamicSize} = cm.ndrange

include("compiler/contract.jl")
include("compiler/pass.jl")

function generate_overdubs(mod, Ctx)
   @eval mod begin
        @inline Cassette.overdub(ctx::$Ctx, ::typeof(groupsize)) = __groupsize(ctx.metadata)
        @inline Cassette.overdub(ctx::$Ctx, ::typeof(__workitems_iterspace)) = workitems(__iterspace(ctx.metadata))

        ###
        # Cassette fixes
        ###
        @inline Cassette.overdub(::$Ctx, ::typeof(Core.kwfunc), f) = return Core.kwfunc(f)
        @inline Cassette.overdub(::$Ctx, ::typeof(Core.apply_type), args...) = return Core.apply_type(args...)
        @inline Cassette.overdub(::$Ctx, ::typeof(StaticArrays.Size), x::Type{<:AbstractArray{<:Any, N}}) where {N} = return StaticArrays.Size(x)

        @inline Cassette.overdub(::$Ctx, ::typeof(+), a::T, b::T) where T<:Union{Float32, Float64} = add_float_contract(a, b)
        @inline Cassette.overdub(::$Ctx, ::typeof(-), a::T, b::T) where T<:Union{Float32, Float64} = sub_float_contract(a, b)
        @inline Cassette.overdub(::$Ctx, ::typeof(*), a::T, b::T) where T<:Union{Float32, Float64} = mul_float_contract(a, b)

        @inline Cassette.overdub(::$Ctx, ::typeof(Base.literal_pow), f::F, x, p) where F = Base.literal_pow(f, x, p)

        @inline Cassette.overdub(::$Ctx, ::typeof(Base.throw_boundserror), args...) = Base.throw_boundserror(args...)
        @inline Cassette.overdub(::$Ctx, ::typeof(Base.Math.throw_exp_domainerror), args...) = Base.Math.throw_exp_domainerror(args...)

        function Cassette.overdub(::$Ctx, ::typeof(:), start::T, step::T, stop::T) where T<:Union{Float16,Float32,Float64}
            lf = (stop-start)/step
            if lf < 0
                len = 0
            elseif lf == 0
                len = 1
            else
                len = round(Int, lf) + 1
                stop′ = start + (len-1)*step
                # if we've overshot the end, subtract one:
                len -= (start < stop < stop′) + (start > stop > stop′)
            end
            Base.steprangelen_hp(T, start, step, 0, len, 1)
        end

        if VERSION >= v"1.5"
            @inline function Cassette.overdub(::$Ctx, ::typeof(Base.Checked.throw_overflowerr_binaryop), op, x, y)
                throw(OverflowError("checked arithmetic: cannot compute"))
            end

            @inline function Cassette.overdub(::$Ctx, ::typeof(Base.Checked.throw_overflowerr_negation), x)
                throw(OverflowError("checked arithmetic: cannot compute -x"))
            end

            @inline function Cassette.overdub(::$Ctx, ::typeof(exponent), x::Union{Float32, Float64})
                T = typeof(x)
                xs = reinterpret(Unsigned, x) & ~Base.sign_mask(T)
                if xs >= Base.exponent_mask(T)
                    throw(DomainError(x, "Cannot be Nan of Inf."))
                end
                k = Int(xs >> Base.significand_bits(T))
                if k == 0 # x is subnormal
                    if xs == 0
                        throw(DomainError(x, "Cannot be subnormal converted to 0."))
                end
                    m = Base.leading_zeros(xs) - Base.exponent_bits(T)
                    k = 1 - m
                end
                return k - Base.exponent_bias(T)
            end
        end
    end
end
