module NVTXExt

import KernelAbstractions as KA
import NVTX

# forwards ranges to Nsight Systems as NVTX ranges, one NVTX domain per domain. NVTX ranges
# annotate host threads; Nsight Systems projects the GPU work launched within them itself.
struct NVTXTracer <: KA.Tracer
    domains::Dict{String, NVTX.Domain}
    lock::ReentrantLock
end
NVTXTracer() = NVTXTracer(Dict{String, NVTX.Domain}(), ReentrantLock())

domain(tracer::NVTXTracer, name::String) =
    @lock tracer.lock get!(() -> NVTX.Domain(name), tracer.domains, name)

# process ranges, rather than push/pop, since they may end on another thread
KA.trace_range_start(tracer::NVTXTracer, label, domain_name) =
    NVTX.range_start(domain(tracer, domain_name); message = label)
KA.trace_range_end(::NVTXTracer, id::NVTX.RangeId) = (NVTX.range_end(id); nothing)
KA.trace_mark(tracer::NVTXTracer, label, domain_name) =
    (NVTX.mark(domain(tracer, domain_name); message = label); nothing)

const TRACER = Ref{NVTXTracer}()

function __init__()
    # only under Nsight, as otherwise every annotation would be wasted work
    if NVTX.isactive()
        TRACER[] = KA.register_tracer!(NVTXTracer())
    end
    return
end

end # module
