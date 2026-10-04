using Printf: @sprintf

###
# Built-in profiler: records the ranges of an expression, and summarizes them
###

# `task` numbers the tasks of a profile in the order they first recorded something, starting
# with 1 for the task that ran `@profile`. A range belongs to the task, and thread, it ended
# on: the range of a `KernelAbstractions.@spawn` starts in the spawning task, and ends in
# the spawned one.
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

# a kernel's execution on the device, in host time
struct ProfileKernel
    name::String
    start::UInt64
    stop::UInt64
    device::String
    task::Int           # whose queue it ran on
    host_timed::Bool    # on a backend without timestamps, by synchronizing
end

struct ProfileTracer <: Tracer
    synchronize::Bool
    device::Bool
    ranges::Vector{ProfileRange}
    markers::Vector{ProfileMarker}
    kernels::Vector{Tuple{Symbol, Int, KernelTimer}}
    tasks::IdDict{Task, Int}
    open::Threads.Atomic{Int}   # ranges started but not ended
    lock::ReentrantLock
end
ProfileTracer(synchronize::Bool, device::Bool = false) = ProfileTracer(
    synchronize, device, ProfileRange[], ProfileMarker[], Tuple{Symbol, Int, KernelTimer}[],
    IdDict{Task, Int}(), Threads.Atomic{Int}(0), ReentrantLock()
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
records_kernels(tracer::ProfileTracer) = tracer.device && in_scope(tracer)

function trace_kernel(tracer::ProfileTracer, label, timer::KernelTimer)
    task = task_number(tracer)
    @lock tracer.lock push!(tracer.kernels, (label, task, timer))
    return nothing
end

# Timestamps only measure intervals on a device, so the kernels are placed on the host's
# clock relative to the first kernel launched on each device, assuming it started when it
# was launched. A kernel can't start before it is launched, which bounds the error.
function resolve_kernels(kernels::Vector{Tuple{Symbol, Int, KernelTimer}})
    first_launched = Dict{Tuple{Any, Int}, KernelTimer}()
    for (_, _, timer) in kernels
        timer.host_timed && continue
        key = (timer.backend, timer.device)
        ref = Base.get(first_launched, key, nothing)
        if ref === nothing || timer.issued < ref.issued
            first_launched[key] = timer
        end
    end
    return map(kernels) do (name, task, timer)
        device = string(nameof(typeof(timer.backend)), " ", timer.device)
        if timer.host_timed
            return ProfileKernel(String(name), timer.start, timer.stop, device, task, true)
        end
        ref = first_launched[(timer.backend, timer.device)]
        offset = KI.elapsed_time(timer.backend, ref.start, timer.start)
        start = max(timer.issued, ref.issued + max(offset, 0))
        return ProfileKernel(String(name), start, start + max(elapsed(timer), 0), device, task, false)
    end
end

# like NVTXT, without domains of their own
profile_name(label, domain) = domain === DEFAULT_DOMAIN ? String(label) : string(domain, ": ", label)

function trace_range_start(tracer::ProfileTracer, label, domain)
    in_scope(tracer) || return nothing
    Threads.atomic_add!(tracer.open, 1)
    return (profile_name(label, domain), time_ns())
end

trace_range_end(::ProfileTracer, ::Nothing) = nothing
function trace_range_end(tracer::ProfileTracer, (name, start))
    range = ProfileRange(name, start, time_ns(), task_number(tracer), Threads.threadid())
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

The ranges, markers and kernels recorded by [`@profile`](@ref KernelAbstractions.@profile).
Shown, it summarizes the time spent per name on the host and on the device or, with
`trace = true`, lists everything in the order it started. `results.ranges`,
`results.markers` and `results.kernels` hold the raw records, with times in nanoseconds
from `time_ns()`.
"""
struct ProfileResults
    start::UInt64
    stop::UInt64
    ranges::Vector{ProfileRange}
    markers::Vector{ProfileMarker}
    kernels::Vector{ProfileKernel}
    trace::Bool
end

"""
    KernelAbstractions.@profile [trace = false] [device = true] [synchronize = false] expr

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
Profiled 3.04 ms, recording 30 ranges and 20 kernels.

Host-side activity:
 Time (%)  Total time  Calls  Avg time  Min time  Max time  Name
 ────────  ──────────  ─────  ────────  ────────  ────────  ────
    6.8 %      206 µs     10   20.6 µs   13.2 µs   77.3 µs  step
    3.5 %      106 µs     10   10.6 µs   5.88 µs   47.1 µs  mul2
    2.7 %     81.5 µs     10   8.15 µs   6.43 µs   20.6 µs  add

Device-side activity:
 Time (%)  Total time  Calls  Avg time  Min time  Max time  Name
 ────────  ──────────  ─────  ────────  ────────  ────────  ────
   51.3 %     1.56 ms     10    156 µs    155 µs    157 µs  add
   45.8 %     1.39 ms     10    139 µs    137 µs    151 µs  mul2
```

On the host, kernel ranges measure the launch. With `device = true`, the default, kernels
are also timed on the device, without synchronizing, on backends that implement
`KernelInterface.record_timestamp`. On other backends, a launch synchronizes its backend to
time the kernel from the host, which is marked with `*`. With `synchronize = true`, kernel
launches also synchronize before their host range ends.

With `trace = true`, the results list every range and kernel in the order it started,
indented by nesting. Kernels are placed on the host's timeline relative to the first kernel
on their device, assuming that one started when it was launched.

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
    options = Dict{Symbol, Any}(:trace => false, :device => true, :synchronize => false)
    for kw in args[1:(end - 1)]
        if Meta.isexpr(kw, :(=)) && haskey(options, kw.args[1])
            options[kw.args[1]] = kw.args[2]
        else
            throw(ArgumentError("KernelAbstractions.@profile: unexpected argument `$kw`; only `trace`, `device` and `synchronize` are accepted"))
        end
    end
    return quote
        $profile(
            () -> $(esc(expr));
            trace = $(esc(options[:trace])), device = $(esc(options[:device])),
            synchronize = $(esc(options[:synchronize]))
        )
    end
end

function profile(f; trace::Bool = false, device::Bool = true, synchronize::Bool = false)
    tracer = register_tracer!(ProfileTracer(synchronize, device))
    task_number(tracer)   # the profiling task is task 1
    start = time_ns()
    try
        with(f, PROFILERS => ProfileTracer[PROFILERS[]; tracer])
    finally
        unregister_tracer!(tracer)
    end
    stop = time_ns()
    open = tracer.open[]
    open > 0 && @warn "$(plural(open, "profiled range")) still open when `@profile` finished; wait for the tasks spawned within it, e.g. with `@sync`"
    ranges, markers, kernels = @lock tracer.lock copy(tracer.ranges), copy(tracer.markers), copy(tracer.kernels)
    # waits for the kernels to complete
    kernels = resolve_kernels(kernels)
    stop = max(stop, maximum(k -> k.stop, kernels; init = stop))
    return ProfileResults(start, stop, ranges, markers, kernels, trace)
end


## report

plural(n, what) = string(n, " ", what, n == 1 ? "" : "s")

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
    nranges, nmarkers, nkernels = length(results.ranges), length(results.markers), length(results.kernels)
    counts = [plural(nranges, "range")]
    nmarkers > 0 && push!(counts, plural(nmarkers, "marker"))
    nkernels > 0 && push!(counts, plural(nkernels, "kernel"))
    println(io, "Profiled ", format_time(total), ", recording ", join(counts, ", ", " and "), ".")
    (nranges == 0 && nmarkers == 0 && nkernels == 0) && return
    println(io)
    if results.trace
        show_trace(io, results)
    else
        show_summary(io, results, total)
    end
    return
end

function summary_table(io::IO, records, total)
    durations = Dict{String, Vector{UInt64}}()
    for record in records
        push!(get!(durations, record.name, UInt64[]), record.stop - record.start)
    end
    rows = sort!(collect(durations); by = kv -> sum(kv[2]), rev = true)
    return print_table(
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

function show_summary(io::IO, results::ProfileResults, total)
    sections = 0
    if !isempty(results.ranges)
        println(io, "Host-side activity:")
        summary_table(io, results.ranges, total)
        sections += 1
    end
    if !isempty(results.kernels)
        sections > 0 && println(io)
        println(io, "Device-side activity:")
        kernels = [
            k.host_timed ? ProfileKernel(k.name * " *", k.start, k.stop, k.device, k.task, true) : k
                for k in results.kernels
        ]
        summary_table(io, kernels, total)
        any(k -> k.host_timed, kernels) &&
            println(io, "\n * timed on the host, as the backend doesn't support timestamps")
        sections += 1
    end
    if !isempty(results.markers)
        sections > 0 && println(io)
        counts = Dict{String, Int}()
        for marker in results.markers
            counts[marker.name] = Base.get(counts, marker.name, 0) + 1
        end
        rows = sort!(collect(counts); by = last, rev = true)
        print_table(io, ["Count", "Marker"], [[string(n), name] for (name, n) in rows])
    end
    return
end

# host records nest on their task, kernels on the queue of their task on their device
location(event::Union{ProfileRange, ProfileMarker}) = "task $(event.task) (thread $(event.thread))"
location(kernel::ProfileKernel) = "$(kernel.device), task $(kernel.task)"

function show_trace(io::IO, results::ProfileResults)
    events = sort!(
        [
            [(r.start, r) for r in results.ranges];
            [(m.time, m) for m in results.markers];
            [(k.start, k) for k in results.kernels]
        ]; by = first
    )
    open = Dict{String, Vector{UInt64}}()   # stop times of the open ranges, per task and queue
    rows = Vector{String}[]
    for (time, event) in events
        stack = get!(open, location(event), UInt64[])
        while !isempty(stack) && last(stack) <= time
            pop!(stack)
        end
        indent = "  "^length(stack)
        start = format_time(time - results.start)
        if event isa ProfileMarker
            push!(rows, [start, "", location(event), indent * "◆ " * event.name])
        else
            push!(stack, event.stop)
            name = event isa ProfileKernel && event.host_timed ? event.name * " *" : event.name
            push!(rows, [start, format_time(event.stop - event.start), location(event), indent * name])
        end
    end
    print_table(io, ["Start", "Duration", "On", "Name"], rows)
    return
end
