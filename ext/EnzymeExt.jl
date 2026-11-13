module EnzymeExt
if isdefined(Base, :get_extension)
    using EnzymeCore
    using EnzymeCore.EnzymeRules
else
    using ..EnzymeCore
    using ..EnzymeCore.EnzymeRules
end

import KernelAbstractions:
    Kernel,
    StaticSize,
    launch_config,
    allocate,
    blocks,
    workitems,
    mkcontext,
    CompilerMetadata,
    CPU,
    GPU,
    argconvert,
    supports_enzyme,
    __fake_compiler_job,
    backend,
    __index_Group_Cartesian,
    __index_Global_Linear,
    __groupsize,
    __groupindex,
    __validindex,
    Backend,
    synchronize

function EnzymeCore.compiler_job_from_backend(
        b::Backend,
        @nospecialize(F::Type),
        @nospecialize(TT::Type),
    )
    error(
        "EnzymeCore.compiler_job_from_backend is not yet implemented for $(typeof(b)), please file an issue.",
    )
end

EnzymeRules.inactive(::Type{StaticSize}, x...) = nothing

# Number of threads a GPU launch runs, for sizing the per-thread tape. `ndrange` is `nothing`
# when the kernel was constructed with a static ndrange and launched without one, so recover
# the thread count from the resolved iteration space instead.
@inline function _ka_nthreads(ndrange, iterspace)
    return ndrange === nothing ?
        length(blocks(iterspace)) * length(workitems(iterspace)) :
        prod(ndrange)
end

@static if isdefined(EnzymeCore, :set_runtime_activity)
    include("EnzymeCore08Ext.jl")
else
    include("EnzymeCore07Ext.jl")
end

# Synchronize rules
# TODO: Right now we do the synchronization as part of the kernel launch in the augmented primal
#       and reverse rules. This is not ideal, as we would want to launch the kernel in the reverse
#       synchronize rule and then synchronize where the launch was. However, with the current
#       kernel semantics this ensures correctness for now.
function EnzymeRules.augmented_primal(
        config,
        func::Const{typeof(synchronize)},
        ::Type{Const{Nothing}},
        backend::T,
    ) where {T <: EnzymeCore.Annotation}
    synchronize(backend.val)
    return AugmentedReturn(nothing, nothing, nothing)
end

function EnzymeRules.reverse(
        config,
        func::Const{typeof(synchronize)},
        ::Type{Const{Nothing}},
        tape,
        backend,
    )
    # noop for now
    return (nothing,)
end

end
