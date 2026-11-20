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
    select_launch,
    CompilerMetadata,
    ConstArgs,
    DynamicCheck,
    CPU,
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

import KernelAbstractions: POCLBackend
import KernelAbstractions: POCL

function EnzymeCore.compiler_job_from_backend(
        ::POCLBackend,
        @nospecialize(F::Type),
        @nospecialize(TT::Type),
    )
    mi = POCL.GPUCompiler.methodinstance(F, TT)
    return POCL.GPUCompiler.CompilerJob(mi, POCL.compiler_config(POCL.device()))
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

# Enzyme caches the values of loops with dynamic trip counts in memory it allocates in device
# code, which the tape points to. POCL kernels allocate from a heap of the launch (see
# `POCL.DeviceHeap`), which has to outlive the reverse pass that reads the caches: allocate
# one per augmented forward launch, sized for `nthreads`, and keep it in the tape. a tape
# without pointers has no caches, and needs none.
_tape_heap(::Backend, TapeType, nthreads) = nothing
_tape_heap(::POCLBackend, TapeType, nthreads) =
    _has_pointers(TapeType) ? POCL.DeviceHeap(POCL.enzyme_heap_size_per_item[] * nthreads) : nothing

_has_pointers(T::Type) = T <: Union{Ptr, Core.LLVMPtr} || any(_has_pointers, fieldtypes(T))

_with_tape_heap(f, ::Nothing) = f()
_with_tape_heap(f, heap::POCL.DeviceHeap) = POCL.with_device_heap(f, heap)

include("EnzymeCore08Ext.jl")

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
