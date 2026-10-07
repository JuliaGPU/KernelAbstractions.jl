## exceptions

# what a kernel reports when one of its work-items throws. it lives in host memory, which the
# CPU device can access directly, and the task that launched the kernel checks it once the
# kernel has completed.
struct ExceptionInfo_st
    # set when a work-item has thrown
    status::Int32
    # claimed by the work-item that reports the exception, see `claim_output`
    output_lock::Int32
    owner::NTuple{4, Int}   # padded: SPIR-V has no vectors of 6 elements, in case of merges

    ExceptionInfo_st() = new(0, 0, (0, 0, 0, 0))
end

# a pointer to one of the fields of the launch's `ExceptionInfo_st`
@inline @generated function exception_field(::Val{field}) where {field}
    T = fieldtype(ExceptionInfo_st, field)
    offset = fieldoffset(ExceptionInfo_st, Base.fieldindex(ExceptionInfo_st, field))
    return :(reinterpret(LLVMPtr{$T, AS.CrossWorkgroup}, kernel_state().exception_info + $offset))
end

# the record is shared by all work-items on the device, so these need device scope, which
# UnsafeAtomics' primitives emit as given (SPIRVIntrinsics' atomics are relaxed)
@inline function atomic_store_device!(ptr::LLVMPtr{Int32, AS.CrossWorkgroup}, val::Int32)
    UnsafeAtomics.Internal.llvm_store!(
        ptr, val, Val(:release), Val(:device), Val(false), Val(sizeof(Int32)), Val(())
    )
    return
end
@inline function atomic_cas_device!(ptr::LLVMPtr{Int32, AS.CrossWorkgroup}, cmp::Int32, val::Int32)
    (; old) = UnsafeAtomics.Internal.llvm_cmpxchg!(
        ptr, cmp, val, Val(:acq_rel), Val(:acquire), Val(:device),
        Val(false), Val(false), Val(sizeof(Int32)), Val(())
    )
    return old
end

# only one work-item reports the exception, or the output of all work-items that throw
# would be interleaved. it can do so over several calls (e.g., a quirk printing the exception,
# and GPUCompiler printing a backtrace). returns 1 when the output was just claimed, 2 when
# it was claimed by an earlier call of this work-item, and 0 when it belongs to another.
@noinline function claim_output()
    lock = exception_field(Val(:output_lock))
    me = (get_global_id(1), get_global_id(2), get_global_id(3), 0)
    state = atomic_cas_device!(lock, Int32(0), Int32(1))
    if state == 0
        unsafe_store!(exception_field(Val(:owner)), me)
        # publish the owner before other work-items compare against it
        atomic_store_device!(lock, Int32(2))
        return 1
    end
    # don't wait for the owner to be published: it may be a work-item that only executes
    # after this one, e.g., in the same loop over a work-group
    return state == 2 && unsafe_load(exception_field(Val(:owner))) == me ? 2 : 0
end

function report_exception(ex)
    claim = claim_output()
    if claim == 1
        SPIRVIntrinsics.@printf(
            "ERROR: %s during kernel execution on work-item (%ld, %ld, %ld).\n", ex, get_global_id(1), get_global_id(2), get_global_id(3)
        )
    end
    if claim != 0
        SPIRVIntrinsics.@printf("Run Julia on debug level 2 for a device stack trace.\n")
    end
    return
end

function report_exception_name(ex)
    claim = claim_output()
    if claim == 1
        SPIRVIntrinsics.@printf(
            "ERROR: %s during kernel execution on work-item (%ld, %ld, %ld).\n", ex, get_global_id(1), get_global_id(2), get_global_id(3)
        )
    end
    if claim != 0
        SPIRVIntrinsics.@printf("Stacktrace:\n")
    end
    return
end

function report_exception_frame(idx, func, file, line)
    if claim_output() != 0
        SPIRVIntrinsics.@printf(" [%d] %s at %s:%d\n", idx, func, file, line)
    end
    return
end

# the work-item that threw stops executing after this
function signal_exception()
    atomic_store_device!(exception_field(Val(:status)), Int32(1))
    return
end

malloc(sz) = C_NULL

report_oom(sz) = return

## kernel state

struct KernelState
    random_seed::UInt32
    # the address of an `ExceptionInfo_st`. not a pointer, as SPIR-V doesn't allow those in
    # kernel arguments that are passed by value.
    exception_info::UInt64
end

@inline @generated kernel_state() = GPUCompiler.kernel_state_value(KernelState)

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
    @eval @inline $name() = additional_arg_value(LLVMPtr{UInt32, AS.Workgroup}, Val($(QuoteNode(name))))
end
