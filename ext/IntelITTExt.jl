module IntelITTExt

using KernelAbstractions: KernelAbstractions, CPU
import KernelAbstractions: profiling_range_active, profiling_range_start, profiling_range_end

import IntelITT

const DOMAINS = Dict{String, IntelITT.Domain}()
const DOMAINS_LOCK = ReentrantLock()

function domain_for(name::AbstractString)
    key = String(name)
    return lock(DOMAINS_LOCK) do
        get!(() -> IntelITT.Domain(key), DOMAINS, key)
    end
end

function profiling_range_active(::CPU; domain = "KernelAbstractions")
    return IntelITT.isactive()
end

function profiling_range_start(::CPU, label; domain = "KernelAbstractions")
    task = IntelITT.Task(domain_for(domain), String(label))
    IntelITT.start(task)
    return task
end

function profiling_range_end(::CPU, id)
    IntelITT.stop(id)
    return nothing
end

end # module
