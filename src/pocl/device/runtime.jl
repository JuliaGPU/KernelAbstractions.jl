signal_exception() = return

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
        push!(state_intr.function_attributes, EnumAttribute(:readnone))
        state_intr
    end
end

# run-time equivalent
function additional_arg_value(state, name)
    return generate_llvmcall(state, Tuple{}) do builder
        T_state = convert(LLVMType, state)

        # get intrinsic
        state_intr = additional_arg_intr(current_module(builder), T_state, name)
        state_intr_ft = state_intr.function_type

        # generate IR
        call!(builder, state_intr_ft, state_intr, Value[], name)
    end
end

for name in [:random_keys, :random_counters]
    @eval @inline @generated $name() =
        additional_arg_value(LLVMPtr{UInt32, AS.Workgroup}, $(String(name)))
end
