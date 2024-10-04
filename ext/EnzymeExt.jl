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

@static if isdefined(EnzymeCore, :set_runtime_activity)
    include("Enzyme013Ext.jl")
else
    include("Enzyme012Ext.jl")
end

# Synchronize rules
# TODO: Right now we do the synchronization as part of the kernel launch in the augmented primal
#       and reverse rules. This is not ideal, as we would want to launch the kernel in the reverse
#       synchronize rule and then synchronize where the launch was. However, with the current
#       kernel semantics this ensures correctness for now.
function EnzymeRules.augmented_primal(
        config::Config,
        func::Const{typeof(synchronize)},
        ::Type{Const{Nothing}},
        backend::T,
    ) where {T <: EnzymeCore.Annotation}
    synchronize(backend.val)
    return AugmentedReturn(nothing, nothing, nothing)
end

function EnzymeRules.reverse(
        config::Config,
        func::Const{typeof(synchronize)},
        ::Type{Const{Nothing}},
        tape,
        backend,
    )
    # noop for now
    return (nothing,)
end

end
