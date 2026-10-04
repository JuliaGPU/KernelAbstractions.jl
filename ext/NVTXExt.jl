module NVTXExt

import KernelAbstractions as KA
import NVTX

# forwards ranges to Nsight Systems as NVTX ranges, one NVTX domain per domain. NVTX ranges
# annotate host threads; Nsight Systems projects the GPU work launched within them itself.
struct NVTXDomain
    domain::NVTX.Domain
    # `Symbol` labels are fixed in the code, so they are registered with NVTX once, which
    # makes recording them cheaper. Other labels are passed as they are.
    strings::IdDict{Symbol, NVTX.StringHandle}
end

struct NVTXTracer <: KA.Tracer
    domains::IdDict{Symbol, NVTXDomain}
    lock::ReentrantLock
end
NVTXTracer() = NVTXTracer(IdDict{Symbol, NVTXDomain}(), ReentrantLock())

function domain(tracer::NVTXTracer, name::Symbol)
    return @lock tracer.lock get!(tracer.domains, name) do
        NVTXDomain(NVTX.Domain(String(name)), IdDict{Symbol, NVTX.StringHandle}())
    end
end

message(::NVTXTracer, d::NVTXDomain, label::String) = label
message(tracer::NVTXTracer, d::NVTXDomain, label::Symbol) =
    @lock tracer.lock get!(() -> NVTX.StringHandle(d.domain, String(label)), d.strings, label)

# process ranges, rather than push/pop, since they may end on another thread
function KA.trace_range_start(tracer::NVTXTracer, label, domain_name)
    d = domain(tracer, domain_name)
    return NVTX.range_start(d.domain; message = message(tracer, d, label))
end
KA.trace_range_end(::NVTXTracer, id::NVTX.RangeId) = (NVTX.range_end(id); nothing)
function KA.trace_mark(tracer::NVTXTracer, label, domain_name)
    d = domain(tracer, domain_name)
    NVTX.mark(d.domain; message = message(tracer, d, label))
    return nothing
end

const TRACER = Ref{NVTXTracer}()

function __init__()
    # only under Nsight, as otherwise every annotation would be wasted work
    if NVTX.isactive()
        TRACER[] = KA.register_tracer!(NVTXTracer())
    end
    return
end

end # module
