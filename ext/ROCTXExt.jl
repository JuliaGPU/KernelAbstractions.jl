module ROCTXExt

import KernelAbstractions as KA
import AMDGPU
using Base.Libc: Libdl

# forwards ranges to rocprof as roctx ranges. like NVTX, roctx annotates host threads, and
# rocprof attributes the GPU work launched within them. roctx has no domains, so ranges in a
# domain other than `:KernelAbstractions` are prefixed with it.
struct ROCTXTracer <: KA.Tracer
    range_start::Ptr{Cvoid}
    range_stop::Ptr{Cvoid}
    mark::Ptr{Cvoid}
end

"""
    ROCTXTracer(library::AbstractString)

A tracer that calls the roctx API in `library`, i.e. `librocprofiler-sdk-roctx` for
`rocprofv3`, or `libroctx64` for the legacy `rocprof`.
"""
function ROCTXTracer(library::AbstractString)
    handle = Libdl.dlopen(library)
    return ROCTXTracer(
        Libdl.dlsym(handle, :roctxRangeStartA), Libdl.dlsym(handle, :roctxRangeStop),
        Libdl.dlsym(handle, :roctxMarkA)
    )
end

# a `Symbol` is passed to C as its name, without allocating
roctx_message(label, domain) = domain === KA.DEFAULT_DOMAIN ? label : string(domain, ": ", label)

# process ranges, rather than push/pop, since they may end on another thread
KA.trace_range_start(tracer::ROCTXTracer, label, domain) =
    ccall(tracer.range_start, UInt64, (Cstring,), roctx_message(label, domain))
KA.trace_range_end(tracer::ROCTXTracer, id::UInt64) =
    (ccall(tracer.range_stop, Cvoid, (UInt64,), id); nothing)
KA.trace_mark(tracer::ROCTXTracer, label, domain) =
    (ccall(tracer.mark, Cvoid, (Cstring,), roctx_message(label, domain)); nothing)

# rocprofv3 loads its tool through rocprofiler-register, and only intercepts the roctx of
# the rocprofiler-sdk; the legacy rocprof loads its tool into HSA, and intercepts libroctx64
function roctx_libraries()
    if haskey(ENV, "ROCP_TOOL_LIBRARIES")
        return ["librocprofiler-sdk-roctx", "libroctx64"]
    elseif haskey(ENV, "HSA_TOOLS_LIB")
        return ["libroctx64", "librocprofiler-sdk-roctx"]
    else
        return String[]
    end
end

function rocm_libdir()
    rocm_path = try
        AMDGPU.ROCmDiscovery.find_roc_path()
    catch
        get(ENV, "ROCM_PATH", "/opt/rocm")
    end
    return joinpath(rocm_path, "lib")
end

const TRACER = Ref{ROCTXTracer}()

function __init__()
    # only under rocprof, as otherwise every annotation would be wasted work
    names = roctx_libraries()
    isempty(names) && return
    library = Libdl.find_library(names, [rocm_libdir()])
    if isempty(library)
        @warn "Running under rocprof, but roctx wasn't found; KernelAbstractions' ranges won't be recorded" names
        return
    end
    TRACER[] = KA.register_tracer!(ROCTXTracer(library))
    return
end

end # module
