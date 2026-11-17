## exceptions

# what a kernel reports when one of its work-items throws. it lives in host memory, which the
# CPU device can access directly, and the task that launched the kernel checks it once the
# kernel has completed.
struct ExceptionInfo_st
    # set by the first work-item that throws, which then records the rest
    status::Int32
    # the throw site, see `GPUCompiler.uses_exception_sites`
    site::Int32
    # the global id of the work-item that threw
    item::NTuple{4, Int}    # padded: SPIR-V has no vectors of 6 elements, in case of merges

    ExceptionInfo_st(status = 0 % Int32, site = 0 % Int32, item = (0, 0, 0, 0)) = new(status, site, item)
end

# a pointer to one of the fields of the launch's `ExceptionInfo_st`
@inline @generated function exception_field(::Val{field}) where {field}
    T = fieldtype(ExceptionInfo_st, field)
    offset = fieldoffset(ExceptionInfo_st, Base.fieldindex(ExceptionInfo_st, field))
    return inline_body(:(reinterpret(LLVMPtr{$T, AS.CrossWorkgroup}, kernel_state().exception_info + $offset)))
end

# device-only functions are overlays, with a host method that throws. sysimage builds compile
# host methods, and a sysimage that references device intrinsics fails to load
# (JuliaGPU/GPUCompiler.jl#611).

# the record is shared by all work-items on the device, so this needs device scope, which
# UnsafeAtomics' primitives emit as given (SPIRVIntrinsics' atomics are relaxed). not a
# compare-and-swap: PoCL's builtin for it returns the old value through a stack slot, which
# PoCL's work-item loops can widen into an array over the work-group
# (JuliaGPU/KernelAbstractions.jl#974).
@device_function @inline function atomic_max_device!(ptr::LLVMPtr{Int32, AS.CrossWorkgroup}, val::Int32)
    return UnsafeAtomics.Internal.llvm_rmw!(
        ptr, Val(:max), val, Val(:acq_rel), Val(:device), Val(false), Val(sizeof(Int32)), Val(())
    )
end

# the work-item that threw stops executing after this. the first one to throw records where;
# the host knows what was thrown there.
#
# nothing reachable from the throw overrides may throw, not even a checked conversion like
# `Int32(0)`: the resulting inference cycle through `throw_inexacterror` taints Base's
# conversions on Julia 1.12+, which then can't be constant-folded (e.g. in `reinterpret`)
@device_function function exception_site(site::Int32)
    if atomic_max_device!(exception_field(Val(:status)), 1 % Int32) == 0
        unsafe_store!(exception_field(Val(:site)), site)
        unsafe_store!(exception_field(Val(:item)), (get_global_id(1), get_global_id(2), get_global_id(3), 0))
    end
    return
end

# these use no device intrinsics, so they can stay host methods. as overlays, inference could
# constant-fold calls to them by running the host method, which throws.
malloc(sz) = C_NULL

report_oom(sz) = return

## kernel state

struct KernelState
    random_seed::UInt32
    # the address of an `ExceptionInfo_st`. not a pointer, as SPIR-V doesn't allow those in
    # kernel arguments that are passed by value.
    exception_info::UInt64
end

@device_function @inline @generated kernel_state() = GPUCompiler.kernel_state_value(KernelState)

## intrinsics for adding and accessing additional kernel arguments

# The amount of local shared memory we need for storing RNG state is determined
# dynamically at kernel launch time, so needs to be passed as additional arguments
# to the kernel.
# We define intrinsics that get transformed into additional kernel arguments which
# then get propagated across function calls to the caller.

function additional_arg_intr(mod::LLVM.Module, T_state, name)
    return get!(mod.functions, "julia.opencl.$name") do
        state_intr = LLVM.Function(mod, "julia.opencl.$name", LLVM.FunctionType(T_state))
        state_intr.memory_effects = MemoryEffects(:none)
        state_intr
    end
end

# run-time equivalent
@llvmgenerated builder function additional_arg_value(::Type{T}, ::Val{name})::T where {T, name}
    state_intr = additional_arg_intr(current_module(builder), convert(LLVMType, T), name)
    call!(builder, state_intr.function_type, state_intr, Value[], String(name))
end

for name in [:random_keys, :random_counters]
    @eval @device_function @inline $name() = additional_arg_value(LLVMPtr{UInt32, AS.Workgroup}, Val($(QuoteNode(name))))
end
