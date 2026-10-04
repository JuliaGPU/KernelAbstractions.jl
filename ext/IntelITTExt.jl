module IntelITTExt

import KernelAbstractions as KA
import IntelITT

# forwards ranges to Intel VTune as ITT tasks, one ITT domain per domain
struct ITTTracer <: KA.Tracer
    domains::IdDict{Symbol, IntelITT.Domain}
    lock::ReentrantLock
end
ITTTracer() = ITTTracer(IdDict{Symbol, IntelITT.Domain}(), ReentrantLock())

domain(tracer::ITTTracer, name::Symbol) =
    @lock tracer.lock get!(() -> IntelITT.Domain(String(name)), tracer.domains, name)

function KA.trace_range_start(tracer::ITTTracer, label, domain_name)
    # overlapped tasks may end on another thread, and needn't nest
    task = IntelITT.Task(domain(tracer, domain_name), String(label))
    IntelITT.start(task)
    return task
end

KA.trace_range_end(::ITTTracer, task::IntelITT.Task) = (IntelITT.stop(task); nothing)

const TRACER = Ref{ITTTracer}()

function __init__()
    # only under a collector, as otherwise every annotation would be wasted work
    if IntelITT.isactive()
        TRACER[] = KA.register_tracer!(ITTTracer())
    end
    return
end

end # module
