struct CompilerMetadata{StaticNDRange, CheckBounds, I, NDRange, Iterspace}
    groupindex::I
    ndrange::NDRange
    iterspace::Iterspace

    # CPU variant
    function CompilerMetadata{NDRange, CB}(idx, ndrange, iterspace) where {NDRange, CB}
        ndrange = cartesian(ndrange)
        return new{NDRange, CB, typeof(idx), typeof(ndrange), typeof(iterspace)}(idx, ndrange, iterspace)
    end

    # GPU variante: index is given implicit
    function CompilerMetadata{NDRange, CB}(ndrange, iterspace) where {NDRange, CB}
        ndrange = cartesian(ndrange)
        return new{NDRange, CB, Nothing, typeof(ndrange), typeof(iterspace)}(nothing, ndrange, iterspace)
    end
end

"""
    cartesian(ndrange)

The object stored as `ndrange` of a kernel context for a launch `ndrange` in any form accepted
by [`partition`](@ref): `CartesianIndices` covering the range, or `nothing` for a static one.
Specialize it together with `partition` for an iteration space of your own, returning an object
that supports `Base.in` for a `CartesianIndex` and [`linear_index`](@ref KernelAbstractions.NDIteration.linear_index).
"""
cartesian(::Nothing) = nothing
cartesian(ci::CartesianIndices) = ci
cartesian(n::Integer) = CartesianIndices((Int(n),))
cartesian(r::AbstractUnitRange) = CartesianIndices((r,))
cartesian(t::Tuple) = CartesianIndices(t)

@inline __iterspace(cm::CompilerMetadata) = cm.iterspace
@inline __groupindex(cm::CompilerMetadata) = cm.groupindex
@inline __groupsize(cm::CompilerMetadata) = size(workitems(__iterspace(cm)))
@inline __dynamic_checkbounds(::CompilerMetadata{NDRange, CB}) where {NDRange, CB} = CB <: DynamicCheck
@inline __ndrange(::CompilerMetadata{NDRange}) where {NDRange <: StaticSize} = CartesianIndices(get(NDRange))
@inline __ndrange(cm::CompilerMetadata{NDRange}) where {NDRange <: DynamicSize} = cm.ndrange
@inline __workitems_iterspace(ctx::CompilerMetadata) = workitems(__iterspace(ctx))

@inline groupsize(ctx::CompilerMetadata) = __groupsize(ctx)
@inline ndrange(ctx::CompilerMetadata) = __ndrange(ctx)
@inline Base.ndims(ctx::CompilerMetadata) = ndims(__iterspace(ctx))

# Adapt the iteration space, which may hold device data in its mapping, and keep the rest
function Adapt.adapt_structure(to, cm::CompilerMetadata{NDRange, CB, I}) where {NDRange, CB, I}
    iterspace = Adapt.adapt(to, cm.iterspace)
    if I === Nothing
        return CompilerMetadata{NDRange, CB}(cm.ndrange, iterspace)
    else
        return CompilerMetadata{NDRange, CB}(cm.groupindex, cm.ndrange, iterspace)
    end
end
