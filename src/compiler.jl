"""
    CompilerMetadata{StaticNDRange, CheckBounds, I, NDRange, Iterspace, Launch, Subgroups}

The hidden context argument of kernels written with [`@kernel`](@ref). The `launch` field
tells the index functions how the backend launched the kernel: `nothing` for a 1-D launch
indexed in `Int`, or a [`LinearLaunch`](@ref) or [`NDLaunch`](@ref). The `subgroups` field
holds the backend's [`SubgroupCapabilities`](@ref) for kernels that use work-group
collectives, or `nothing`.
"""
struct CompilerMetadata{StaticNDRange, CheckBounds, I, NDRange, Iterspace, Launch, Subgroups}
    groupindex::I
    ndrange::NDRange
    iterspace::Iterspace
    launch::Launch
    subgroups::Subgroups

    # CPU variant
    function CompilerMetadata{NDRange, CB}(idx, ndrange, iterspace) where {NDRange, CB}
        ndrange = cartesian(ndrange)
        return new{NDRange, CB, typeof(idx), typeof(ndrange), typeof(iterspace), Nothing, Nothing}(idx, ndrange, iterspace, nothing, nothing)
    end

    # GPU variante: index is given implicit
    function CompilerMetadata{NDRange, CB}(ndrange, iterspace; launch = nothing, subgroups = nothing) where {NDRange, CB}
        ndrange = cartesian(ndrange)
        return new{NDRange, CB, Nothing, typeof(ndrange), typeof(iterspace), typeof(launch), typeof(subgroups)}(nothing, ndrange, iterspace, launch, subgroups)
    end
end

# `CartesianIndices` covering a launch `ndrange` (any form accepted by `partition`).
cartesian(::Nothing) = nothing
cartesian(ci::CartesianIndices) = ci
cartesian(n::Integer) = CartesianIndices((Int(n),))
cartesian(r::AbstractUnitRange) = CartesianIndices((r,))
cartesian(t::Tuple) = CartesianIndices(t)

@inline __iterspace(cm::CompilerMetadata) = cm.iterspace
@inline __groupindex(cm::CompilerMetadata) = cm.groupindex
@inline __launch(cm::CompilerMetadata) = cm.launch
@inline __subgroups(cm::CompilerMetadata) = cm.subgroups
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
        return CompilerMetadata{NDRange, CB}(cm.ndrange, iterspace; cm.launch, cm.subgroups)
    else
        return CompilerMetadata{NDRange, CB}(cm.groupindex, cm.ndrange, iterspace)
    end
end
