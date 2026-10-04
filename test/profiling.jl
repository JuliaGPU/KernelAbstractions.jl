# records the ranges and markers it is given
struct RecordingTracer <: KernelAbstractions.Tracer
    events::Vector{Any}
    lock::ReentrantLock
end
RecordingTracer() = RecordingTracer([], ReentrantLock())
function KernelAbstractions.trace_range_start(t::RecordingTracer, label, domain)
    @lock t.lock push!(t.events, (:start, label, domain))
    return label
end
KernelAbstractions.trace_range_end(t::RecordingTracer, id) =
    @lock t.lock push!(t.events, (:end, id))
KernelAbstractions.trace_mark(t::RecordingTracer, label, domain) =
    @lock t.lock push!(t.events, (:mark, label, domain))

function with_tracer(f, tracer = RecordingTracer())
    KernelAbstractions.register_tracer!(tracer)
    try
        f(tracer)
    finally
        KernelAbstractions.unregister_tracer!(tracer)
    end
    return tracer
end

@kernel function profiling_fill!(A, x)
    I = @index(Global)
    @inbounds A[I] = x
end

function profiling_testsuite(Backend, AT)
    backend = Backend()

    # launches work whether or not a profiler listens
    A = AT(zeros(Float32, 64))
    profiling_fill!(backend)(A, 1.0f0; ndrange = length(A))
    synchronize(backend)
    @test all(Array(A) .== 1)

    # and are named after the kernel
    tracer = with_tracer() do tracer
        @profiling_range "step" profiling_fill!(backend)(A, 2.0f0; ndrange = length(A))
        synchronize(backend)
    end
    @test all(Array(A) .== 2)
    @test tracer.events == [
        (:start, "step", "KernelAbstractions"),
        (:start, "profiling_fill!", "KernelAbstractions"), (:end, "profiling_fill!"),
        (:end, "step"),
    ]

    return
end
