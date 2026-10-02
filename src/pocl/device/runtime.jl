## exceptions

# what a kernel reports when one of its work-items throws. it lives in host memory, which the
# CPU device can access directly, and the task that launched the kernel checks it once the
# kernel has completed.
struct ExceptionInfo_st
    # set when a work-item has thrown
    status::Int32

    ExceptionInfo_st() = new(0)
end

# a pointer to one of the fields of the launch's `ExceptionInfo_st`
@inline @generated function exception_field(::Val{field}) where {field}
    T = fieldtype(ExceptionInfo_st, field)
    offset = fieldoffset(ExceptionInfo_st, Base.fieldindex(ExceptionInfo_st, field))
    return :(reinterpret(LLVMPtr{$T, AS.CrossWorkgroup}, kernel_state().exception_info + $offset))
end

# the record is shared by all work-items on the device, and SPIRVIntrinsics' atomics only
# have work-group scope
@inline atomic_store_device!(ptr::LLVMPtr{Int32, AS.CrossWorkgroup}, val::Int32) =
    @builtin_ccall(
    "__spirv_AtomicStore", Cvoid, (LLVMPtr{Int32, AS.CrossWorkgroup}, UInt32, UInt32, Int32),
    ptr, UInt32(Scope.Device), UInt32(MemorySemantics.CrossWorkgroupMemory | MemorySemantics.Release), val
)

# the work-item that threw stops executing after this
function signal_exception()
    atomic_store_device!(exception_field(Val(:status)), Int32(1))
    return
end

malloc(sz) = C_NULL

report_oom(sz) = return

import SPIRVIntrinsics: get_global_id

function report_exception(ex)
    SPIRVIntrinsics.@printf(
        "ERROR: a %s was thrown during kernel execution on thread (%d, %d, %d).\n",
        ex, get_global_id(UInt32(1)), get_global_id(UInt32(2)), get_global_id(UInt32(3))
    )
    return
end

function report_exception_name(ex)
    SPIRVIntrinsics.@printf(
        "ERROR: a %s was thrown during kernel execution on thread (%d, %d, %d).\n",
        ex, get_global_id(UInt32(1)), get_global_id(UInt32(2)), get_global_id(UInt32(3))
    )
    SPIRVIntrinsics.@printf("Stacktrace:\n")
    return
end

function report_exception_frame(idx, func, file, line)
    SPIRVIntrinsics.@printf(" [%d] %s at %s:%d\n", idx, func, file, line)
    return
end

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
    state_intr = if haskey(functions(mod), "julia.opencl.$name")
        functions(mod)["julia.opencl.$name"]
    else
        LLVM.Function(mod, "julia.opencl.$name", LLVM.FunctionType(T_state))
    end
    push!(function_attributes(state_intr), EnumAttribute("readnone", 0))

    return state_intr
end

# run-time equivalent
function additional_arg_value(state, name)
    return @dispose ctx = Context() begin
        T_state = convert(LLVMType, state)

        # create function
        llvm_f, _ = create_function(T_state)
        mod = LLVM.parent(llvm_f)

        # get intrinsic
        state_intr = additional_arg_intr(mod, T_state, name)
        state_intr_ft = function_type(state_intr)

        # generate IR
        @dispose builder = IRBuilder() begin
            entry = BasicBlock(llvm_f, "entry")
            position!(builder, entry)

            val = call!(builder, state_intr_ft, state_intr, Value[], name)

            ret!(builder, val)
        end

        call_function(llvm_f, state)
    end
end

for name in [:random_keys, :random_counters]
    @eval @inline @generated $name() =
        additional_arg_value(LLVMPtr{UInt32, AS.Workgroup}, $(String(name)))
end
