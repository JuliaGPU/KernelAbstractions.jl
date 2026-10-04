using Printf: @sprintf
using ScopedValues: ScopedValue, with

###
# Built-in profiler: records the ranges of an expression, and summarizes them
###

# `task` numbers the tasks of a profile in the order they first recorded something, starting
# with 1 for the task that ran `@profile`; `thread` is the thread a range started on
struct ProfileRange
    name::String
    start::UInt64
    stop::UInt64
    task::Int
    thread::Int
end

struct ProfileMarker
    name::String
    time::UInt64
    task::Int
    thread::Int
end

struct ProfileTracer <: Tracer
    synchronize::Bool
    ranges::Vector{ProfileRange}
    markers::Vector{ProfileMarker}
    tasks::IdDict{Task, Int}
    open::Threads.Atomic{Int}   # ranges started but not ended
    lock::ReentrantLock
end
ProfileTracer(synchronize::Bool) = ProfileTracer(
    synchronize, ProfileRange[], ProfileMarker[], IdDict{Task, Int}(), Threads.Atomic{Int}(0),
    ReentrantLock()
)

# The profilers whose expression the current task is running, directly or in a task it
# spawned: tracers are global, but `@profile` only records the tasks of its expression.
const PROFILERS = ScopedValue{Vector{ProfileTracer}}(ProfileTracer[])

in_scope(tracer::ProfileTracer) = any(t -> t === tracer, PROFILERS[])

function task_number(tracer::ProfileTracer)
    task = current_task()
    return @lock tracer.lock get!(tracer.tasks, task, length(tracer.tasks) + 1)
end

synchronizes_launches(tracer::ProfileTracer) = tracer.synchronize && in_scope(tracer)

# like NVTXT, without domains of their own
profile_name(label, domain) = domain == "KernelAbstractions" ? label : string(domain, ": ", label)

function trace_range_start(tracer::ProfileTracer, label, domain)
    in_scope(tracer) || return nothing
    Threads.atomic_add!(tracer.open, 1)
    return (profile_name(label, domain), time_ns(), task_number(tracer), Threads.threadid())
end

trace_range_end(::ProfileTracer, ::Nothing) = nothing
function trace_range_end(tracer::ProfileTracer, (name, start, task, thread))
    range = ProfileRange(name, start, time_ns(), task, thread)
    @lock tracer.lock push!(tracer.ranges, range)
    Threads.atomic_sub!(tracer.open, 1)
    return nothing
end

function trace_mark(tracer::ProfileTracer, label, domain)
    in_scope(tracer) || return nothing
    marker = ProfileMarker(profile_name(label, domain), time_ns(), task_number(tracer), Threads.threadid())
    @lock tracer.lock push!(tracer.markers, marker)
    return nothing
end

"""
    ProfileResults

The ranges and markers recorded by [`@profile`](@ref KernelAbstractions.@profile). Shown,
it summarizes the time spent per range name or, with `trace = true`, lists the ranges in
the order they started. `results.ranges` and `results.markers` hold the raw records, with
times in nanoseconds from `time_ns()`.
"""
struct ProfileResults
    start::UInt64
    stop::UInt64
    ranges::Vector{ProfileRange}
    markers::Vector{ProfileMarker}
    trace::Bool
end

"""
    KernelAbstractions.@profile [trace = false] [synchronize = true] expr

Run `expr`, recording the ranges of [`@profiling_range`](@ref), the markers of
[`profiling_mark`](@ref) and the kernel launches within it, and return a
[`ProfileResults`](@ref KernelAbstractions.ProfileResults) that summarizes them:

```julia-repl
julia> KernelAbstractions.@profile for i in 1:10
           @profiling_range "step" begin
               mul2(backend)(A; ndrange = length(A))
               add(backend)(A, B; ndrange = length(A))
           end
       end
Profiled 6.04 ms, recording 30 ranges.

 Time (%)  Total time  Calls  Avg time  Min time  Max time  Name
 ────────  ──────────  ─────  ────────  ────────  ────────  ────
   92.6 %     5.59 ms     10    559 µs    298 µs    2.9 ms  step
   57.6 %     3.48 ms     10    348 µs    109 µs   2.48 ms  mul2
   37.6 %     2.27 ms     10    227 µs    179 µs    400 µs  add
```

With `synchronize = true`, the default, kernel launches synchronize their backend before
their range ends, so that on GPU backends they measure the kernel's execution instead of
its launch. This serializes the host with the device, as `CUDA_LAUNCH_BLOCKING=1` does.

With `trace = true`, the results list every range in the order it started instead,
indented by nesting on its task.

Only the task running `expr` and the tasks it spawns (with `KernelAbstractions.@spawn`,
`Threads.@spawn` or `@async`) are recorded, so that other tasks, including other
`@profile`s, don't show up in the results. Wait for spawned tasks within `expr`, e.g. with
`@sync`: what a task records after `expr` has returned is lost, and `@profile` warns about
ranges that were still open.

Other registered tracers, e.g. NVTX under Nsight Systems, record the ranges of all tasks.
"""
macro profile(args...)
    isempty(args) && throw(ArgumentError("KernelAbstractions.@profile needs an expression to profile"))
    expr = args[end]
    trace, synchronize = false, true
    for kw in args[1:(end - 1)]
        if Meta.isexpr(kw, :(=)) && kw.args[1] === :trace
            trace = kw.args[2]
        elseif Meta.isexpr(kw, :(=)) && kw.args[1] === :synchronize
            synchronize = kw.args[2]
        else
            throw(ArgumentError("KernelAbstractions.@profile: unexpected argument `$kw`; only `trace = ...` and `synchronize = ...` are accepted"))
        end
    end
    return quote
        $profile(() -> $(esc(expr)); trace = $(esc(trace)), synchronize = $(esc(synchronize)))
    end
end

function profile(f; trace::Bool = false, synchronize::Bool = true)
    tracer = register_tracer!(ProfileTracer(synchronize))
    task_number(tracer)   # the profiling task is task 1
    start = time_ns()
    try
        with(f, PROFILERS => ProfileTracer[PROFILERS[]; tracer])
    finally
        unregister_tracer!(tracer)
    end
    stop = time_ns()
    open = tracer.open[]
    open > 0 && @warn "$open profiled ranges were still open when `@profile` finished; wait for the tasks spawned within it, e.g. with `@sync`"
    return @lock tracer.lock ProfileResults(start, stop, copy(tracer.ranges), copy(tracer.markers), trace)
end


## report

function format_time(ns::Real)
    # switch units where three significant digits would round up to the next one
    ns < 999.5 && return @sprintf("%.0f ns", ns)
    ns < 999.5e3 && return @sprintf("%.3g µs", ns / 1.0e3)
    ns < 999.5e6 && return @sprintf("%.3g ms", ns / 1.0e6)
    return @sprintf("%.3g s", ns / 1.0e9)
end

# columns of strings, the last one left-aligned
function print_table(io::IO, header::Vector{String}, rows::Vector{Vector{String}})
    widths = [maximum(textwidth, [h; getindex.(rows, i)]) for (i, h) in enumerate(header)]
    function print_row(row)
        print(io, " ")
        for (i, (cell, width)) in enumerate(zip(row, widths))
            if i == length(row)
                print(io, cell)
            else
                print(io, lpad(cell, width), "  ")
            end
        end
        return println(io)
    end
    print_row(header)
    print_row(["─"^w for w in widths])
    foreach(print_row, rows)
    return
end

function Base.show(io::IO, ::MIME"text/plain", results::ProfileResults)
    total = results.stop - results.start
    nranges, nmarkers = length(results.ranges), length(results.markers)
    print(io, "Profiled ", format_time(total), ", recording ", nranges, nranges == 1 ? " range" : " ranges")
    nmarkers > 0 && print(io, " and ", nmarkers, nmarkers == 1 ? " marker" : " markers")
    println(io, ".")
    (nranges == 0 && nmarkers == 0) && return
    println(io)
    if results.trace
        show_trace(io, results)
    else
        show_summary(io, results, total)
    end
    return
end

function show_summary(io::IO, results::ProfileResults, total)
    if !isempty(results.ranges)
        durations = Dict{String, Vector{UInt64}}()
        for range in results.ranges
            push!(get!(durations, range.name, UInt64[]), range.stop - range.start)
        end
        rows = sort!(collect(durations); by = kv -> sum(kv[2]), rev = true)
        print_table(
            io, ["Time (%)", "Total time", "Calls", "Avg time", "Min time", "Max time", "Name"],
            [
                [
                    @sprintf("%.1f %%", 100 * sum(ds) / total), format_time(sum(ds)), string(length(ds)),
                    format_time(sum(ds) / length(ds)), format_time(minimum(ds)), format_time(maximum(ds)),
                    name,
                ] for (name, ds) in rows
            ]
        )
    end
    if !isempty(results.markers)
        isempty(results.ranges) || println(io)
        counts = Dict{String, Int}()
        for marker in results.markers
            counts[marker.name] = Base.get(counts, marker.name, 0) + 1
        end
        rows = sort!(collect(counts); by = last, rev = true)
        print_table(io, ["Count", "Marker"], [[string(n), name] for (name, n) in rows])
    end
    return
end

location(event) = "task $(event.task) (thread $(event.thread))"

function show_trace(io::IO, results::ProfileResults)
    # the nesting depth of each range among the ranges on its task
    events = sort!(
        [
            [(r.start, r) for r in results.ranges];
            [(m.time, m) for m in results.markers]
        ]; by = first
    )
    open = Dict{Int, Vector{UInt64}}()   # stop times of the open ranges, per task
    rows = Vector{String}[]
    for (time, event) in events
        stack = get!(open, event.task, UInt64[])
        while !isempty(stack) && last(stack) <= time
            pop!(stack)
        end
        indent = "  "^length(stack)
        if event isa ProfileRange
            push!(stack, event.stop)
            push!(rows, [format_time(time - results.start), format_time(event.stop - event.start), location(event), indent * event.name])
        else
            push!(rows, [format_time(time - results.start), "", location(event), indent * "◆ " * event.name])
        end
    end
    print_table(io, ["Start", "Duration", "On", "Name"], rows)
    return
end
