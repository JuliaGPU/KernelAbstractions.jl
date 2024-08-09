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

@inline __iterspace(cm::CompilerMetadata) = cm.iterspace
@inline __groupindex(cm::CompilerMetadata) = cm.groupindex
@inline __groupsize(cm::CompilerMetadata) = size(workitems(__iterspace(cm)))
@inline __dynamic_checkbounds(::CompilerMetadata{NDRange, CB}) where {NDRange, CB} = CB <: DynamicCheck
@inline __ndrange(::CompilerMetadata{NDRange}) where {NDRange <: StaticSize} = CartesianIndices(get(NDRange))
@inline __ndrange(cm::CompilerMetadata{NDRange}) where {NDRange <: DynamicSize} = cm.ndrange
@inline __workitems_iterspace(ctx) = workitems(__iterspace(ctx))

@inline groupsize(ctx) = __groupsize(ctx)
@inline ndrange(ctx) = __ndrange(ctx)
