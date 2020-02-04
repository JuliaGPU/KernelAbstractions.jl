struct CompilerMetadata{StaticWorkgroupSize, StaticNDRange, CheckBounds, I, NDRange, WorkgroupSize}
    groupindex::I
    ndrange::NDRange
    workgroupsize::WorkgroupSize

    # CPU variant
    function CompilerMetadata{WorkgroupSize, NDRange, CB}(idx, ndrange, workgroupsize) where {WorkgroupSize, NDRange, CB}
        if ndrange !== nothing
            ndrange = CartesianIndices(ndrange)
        end
        return new{WorkgroupSize, NDRange, CB, typeof(idx), typeof(ndrange), typeof(workgroupsize)}(idx, ndrange, workgroupsize)
    end

    # GPU variante index is given implicit
    function CompilerMetadata{WorkgroupSize, NDRange, CB}(ndrange, workgroupsize) where {WorkgroupSize, NDRange, CB}
        if ndrange !== nothing
            ndrange = CartesianIndices(ndrange)
        end
        return new{WorkgroupSize, NDRange, CB, Nothing, typeof(ndrange), typeof(workgroupsize)}(nothing, ndrange, workgroupsize)
    end
end

@inline __groupsize(::CompilerMetadata{WorkgroupSize}) where {WorkgroupSize<:StaticSize} = get(WorkgroupSize)[1]
@inline __groupsize(::CompilerMetadata{WorkgroupSize}) where {WorkgroupSize<:DynamicSize} = cm.workgroupsize
@inline __groupindex(cm::CompilerMetadata) = cm.groupindex
@inline __dynamic_checkbounds(::CompilerMetadata{WorkgroupSize, NDRange, CB}) where {WorkgroupSize, NDRange, CB} = CB
@inline __ndrange(cm::CompilerMetadata{WorkgroupSize, NDRange}) where {WorkgroupSize, NDRange<:StaticSize}  = CartesianIndices(get(NDRange))
@inline __ndrange(cm::CompilerMetadata{WorkgroupSize, NDRange}) where {WorkgroupSize, NDRange<:DynamicSize} = cm.ndrange

include("compiler/contract.jl")
include("compiler/pass.jl")

function generate_overdubs(Ctx)
   @eval begin
        @inline Cassette.overdub(ctx::$Ctx, ::typeof(groupsize)) = __groupsize(ctx.metadata)

        ###
        # Cassette fixes
        ###
        @inline Cassette.overdub(::$Ctx, ::typeof(Core.kwfunc), f) = return Core.kwfunc(f)
        @inline Cassette.overdub(::$Ctx, ::typeof(Core.apply_type), args...) = return Core.apply_type(args...)
        @inline Cassette.overdub(::$Ctx, ::typeof(StaticArrays.Size), x::Type{<:AbstractArray{<:Any, N}}) where {N} = return StaticArrays.Size(x)

        @inline Cassette.overdub(::$Ctx, ::typeof(+), a::T, b::T) where T<:Union{Float32, Float64} = add_float_contract(a, b)
        @inline Cassette.overdub(::$Ctx, ::typeof(-), a::T, b::T) where T<:Union{Float32, Float64} = sub_float_contract(a, b)
        @inline Cassette.overdub(::$Ctx, ::typeof(*), a::T, b::T) where T<:Union{Float32, Float64} = mul_float_contract(a, b)

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
    end
end
