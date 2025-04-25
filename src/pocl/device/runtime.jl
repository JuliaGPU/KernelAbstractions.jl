# reset the runtime cache from global scope, so that any change triggers recompilation
GPUCompiler.reset_runtime()

signal_exception() = return

malloc(sz) = C_NULL

report_oom(sz) = return

import SPIRVIntrinsics: get_global_id

function report_exception(ex)
    SPIRVIntrinsics.@printf(
        "ERROR: a %s was thrown during kernel execution on thread (%d, %d, %d).\n",
        ex, get_global_id(UInt32(0)), get_global_id(UInt32(1)), get_global_id(UInt32(2))
    )
    return
end

function report_exception_name(ex)
    SPIRVIntrinsics.@printf(
        "ERROR: a %s was thrown during kernel execution on thread (%d, %d, %d).\n",
        ex, get_global_id(UInt32(0)), get_global_id(UInt32(1)), get_global_id(UInt32(2))
    )
    SPIRVIntrinsics.@printf("Stacktrace:\n")
    return
end

function report_exception_frame(idx, func, file, line)
    SPIRVIntrinsics.@printf(" [%d] %s at %s:%d\n", idx, func, file, line)
    return
end
