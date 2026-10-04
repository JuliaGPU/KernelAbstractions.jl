###
# Profiler integration
#
# Tracing profilers (Nsight Systems via NVTX, VTune via ITT, rocprof via roctx, ...) record
# ranges on the host threads of the process, and correlate the device work launched within
# them themselves. Which profiler records a range thus depends on what the process runs
# under, not on the backend: ranges go to every registered tracer, for every backend. With
# none registered, an annotation costs one atomic load.
###

"""
    Tracer

Abstract supertype for profilers that record named ranges on the host. Register an
instance with [`register_tracer!`](@ref) to receive the ranges and markers of
[`@profiling_range`](@ref) and [`profiling_mark`](@ref), and of kernel launches.

Subtypes implement

    trace_range_start(tracer, label::Label, domain::Symbol) -> id
    trace_range_end(tracer, id)
    trace_mark(tracer, label::Label, domain::Symbol)     # optional
    synchronizes_launches(tracer)::Bool                     # optional, default `false`

If `synchronizes_launches` is `true`, kernel launches synchronize their backend before their
range ends, so that the range measures the kernel's execution rather than its launch.

A `Label` is a `Symbol` for labels that are fixed in the code: literals in
[`@profiling_range`](@ref), kernel names and `@spawn` call sites. As there are only so many
of those, tracers may cache what they derive from a `Symbol` label, e.g. a registered
string, keyed by its identity. Labels computed at run time are `String`s, which tracers
should not cache. The default domain is `:KernelAbstractions`.

Ranges may end on a different thread than they started on, and may overlap without nesting,
so implementations should use the profiler's start/end API (e.g. `nvtxRangeStartEx`) rather
than a thread-local push/pop stack.

A tracer for a profiler that may not be attached should only be registered when it is, as
registering any tracer makes every annotation do work.
"""
abstract type Tracer end

const Label = Union{Symbol, String}
const DEFAULT_DOMAIN = :KernelAbstractions

# literals are made `Symbol`s by the macros, anything else stays a `String`
as_label(label::Symbol) = label
as_label(label::AbstractString) = String(label)
as_label(label) = string(label)
as_domain(domain::Symbol) = domain
as_domain(domain) = Symbol(domain)

function trace_range_start end
function trace_range_end end
trace_mark(::Tracer, label, domain) = nothing
synchronizes_launches(::Tracer) = false

# copy-on-write, so that checking for tracers is a single atomic load
mutable struct Tracers
    @atomic tracers::Vector{Tracer}
end
const TRACERS = Tracers(Tracer[])
const TRACERS_LOCK = ReentrantLock()

tracers() = @atomic :acquire TRACERS.tracers

"""
    register_tracer!(tracer::Tracer) -> tracer

Forward profiler ranges and markers to `tracer`, in addition to the tracers that are
registered already.
"""
function register_tracer!(tracer::Tracer)
    @lock TRACERS_LOCK begin
        current = tracers()
        tracer in current || @atomic :release TRACERS.tracers = Tracer[current; tracer]
    end
    return tracer
end

"""
    unregister_tracer!(tracer::Tracer)

Stop forwarding profiler ranges and markers to `tracer`. Ranges that are open keep going to
the tracers that were registered when they started.
"""
function unregister_tracer!(tracer::Tracer)
    @lock TRACERS_LOCK begin
        @atomic :release TRACERS.tracers = filter(t -> t !== tracer, tracers())
    end
    return nothing
end

"""
    profiling_active()::Bool

Whether a profiler is listening, i.e. whether a [`Tracer`](@ref) is registered. Check this
before doing work that only serves annotations.
"""
profiling_active() = !isempty(tracers())

struct ProfilingRange
    tracers::Vector{Tracer}
    ids::Vector{Any}
end

"""
    profiling_range_start(label; domain = :KernelAbstractions)

Start a range named `label` in `domain`, and return a handle for
[`profiling_range_end`](@ref). Returns `nothing` if no profiler is listening.

Prefer [`@profiling_range`](@ref), which ends the range even if an exception is thrown. Use
these for ranges that don't follow the structure of the code.
"""
function profiling_range_start(label; domain = DEFAULT_DOMAIN)
    current = tracers()
    isempty(current) && return nothing
    label, domain = as_label(label), as_domain(domain)
    return ProfilingRange(current, Any[trace_range_start(t, label, domain) for t in current])
end

"""
    profiling_range_end(range)

End a range started with [`profiling_range_start`](@ref).
"""
function profiling_range_end(range::ProfilingRange)
    for (tracer, id) in zip(range.tracers, range.ids)
        trace_range_end(tracer, id)
    end
    return nothing
end
profiling_range_end(::Nothing) = nothing

synchronizes_launches(range::ProfilingRange) = any(synchronizes_launches, range.tracers)
synchronizes_launches(::Nothing) = false

"""
    profiling_mark(label; domain = :KernelAbstractions)

Record an instantaneous marker named `label` in `domain`.
"""
# the check is inlined into the caller, so that a marker costs nothing when nobody listens
@inline profiling_mark(label; domain = DEFAULT_DOMAIN) =
    profiling_active() ? record_mark(label, domain) : nothing

@noinline function record_mark(label, domain)
    current = tracers()
    label, domain = as_label(label), as_domain(domain)
    for tracer in current
        trace_mark(tracer, label, domain)
    end
    return nothing
end

"""
    @profiling_range label [domain = :KernelAbstractions] expr

Evaluate `expr` inside a profiler range named `label`, and return its value. `label` is
only evaluated if a profiler is listening (see [`profiling_active`](@ref)), so it can be
built with string interpolation at no cost to unprofiled runs. The range is ended if `expr`
throws. Assignments in `expr` are visible after the macro, as with `@time`.

`expr` is compiled twice, for when a profiler listens and for when none does, so that the
latter costs no more than a check. It therefore can't define labels: `@goto` and `@label`
are not supported in `expr`.

```julia
@profiling_range "volume integral" begin
    volume_integral!(du, u, backend)
end

@profiling_range "volume integral" domain = "Trixi" begin
    volume_integral!(du, u, backend)
end
```

Ranges are recorded on the host, by whichever profiler the process runs under; device work
launched within a range is attributed to it by profilers that correlate the two, such as
Nsight Systems. Kernel launches are annotated with the name of the kernel automatically.
"""
macro profiling_range(label, args...)
    isempty(args) && throw(ArgumentError("@profiling_range needs an expression to evaluate"))
    expr = args[end]
    domain = QuoteNode(DEFAULT_DOMAIN)
    for kw in args[1:(end - 1)]
        if Meta.isexpr(kw, :(=)) && kw.args[1] === :domain
            domain = kw.args[2]
        else
            throw(ArgumentError("@profiling_range: unexpected argument `$kw`; only `domain = ...` is accepted"))
        end
    end
    id = gensym(:id)
    # unlike `try`, `tryfinally` doesn't introduce a scope
    traced = Expr(:tryfinally, esc(expr), :($profiling_range_end($id)))
    # Entering the exception handler that ends the range costs more than checking for a
    # profiler, so it is only entered when one listens, at the price of compiling `expr`
    # twice.
    return quote
        if $profiling_active()
            local $id = $profiling_range_start($(literal(label)); domain = $(literal(domain)))
            $traced
        else
            $(esc(expr))
        end
    end
end

# a string literal is a `Symbol` label, fixed in the code; anything else is evaluated
literal(x::String) = QuoteNode(Symbol(x))
literal(x::QuoteNode) = x
literal(x) = esc(x)

# the name kernel launches are annotated with: `@kernel function f` compiles to `gpu_f`
# (computed once per kernel function type)
const KERNEL_LABELS = IdDict{Any, Symbol}()
const KERNEL_LABELS_LOCK = ReentrantLock()
function kernel_label(f)
    return @lock KERNEL_LABELS_LOCK get!(KERNEL_LABELS, typeof(f)) do
        f isa Function || return Symbol(typeof(f))
        name = string(nameof(f))
        return Symbol(startswith(name, "gpu_") ? name[5:end] : name)
    end
end


## NVTXT

"""
    NVTXTTracer(path::AbstractString)
    NVTXTTracer(io::IO)

A [`Tracer`](@ref) that writes ranges and markers in the NVTXT text format, which NVIDIA
Nsight Systems imports with `ImportNvtxt`:

```
ImportNvtxt --cmd create --nvtxt ka-1234.nvtxt -o report.nsys-rep
```

This needs no profiler at run time, e.g. to trace the CPU backend on a machine without
Nsight Systems. Set `JULIA_KA_NVTXT` to start one when KernelAbstractions is loaded: to a
path, in which `%p` is replaced by the process id, or to `1` for `ka-%p.nvtxt` in the
working directory. Otherwise, register one with [`register_tracer!`](@ref), and `close` it
after unregistering it to flush the file.

Records are written when a range ends, as a single line, so that threads don't interleave.
Ranges in a `domain` other than `:KernelAbstractions` are prefixed with it.
"""
struct NVTXTTracer{IO_ <: IO} <: Tracer
    io::IO_
    lock::ReentrantLock
    record::Vector{UInt8}                               # formatted under the lock
    messages::IdDict{Tuple{Symbol, Symbol}, String}     # of `Symbol` labels
end

function NVTXTTracer(io::IO)
    pid = getpid()
    print(
        io, """
        SetFileDisplayName, KernelAbstractions
        @RangeStartEnd, Start, End, ThreadId, Message
        ProcessId = $pid
        CategoryId = 1
        Color = Blue
        TimeBase = Manual
        @Marker, Time, ThreadId, Message
        ProcessId = $pid
        CategoryId = 1
        Color = Blue
        TimeBase = Manual
        """
    )
    return NVTXTTracer(io, ReentrantLock(), sizehint!(UInt8[], 256), IdDict{Tuple{Symbol, Symbol}, String}())
end
NVTXTTracer(path::AbstractString) = NVTXTTracer(open(path, "w"))

Base.close(tracer::NVTXTTracer) = @lock tracer.lock close(tracer.io)

struct NVTXTRange
    start::UInt64
    thread::Int
    message::String
end

# the message is a quoted string, and a record a line
function nvtxt_message(label, domain::Symbol)
    message = domain === DEFAULT_DOMAIN ? String(label) : string(domain, ": ", label)
    return replace(message, '"' => '\'', '\n' => ' ', '\r' => ' ')
end
nvtxt_message(tracer::NVTXTTracer, label::String, domain::Symbol) = nvtxt_message(label, domain)
nvtxt_message(tracer::NVTXTTracer, label::Symbol, domain::Symbol) =
    @lock tracer.lock get!(() -> nvtxt_message(label, domain), tracer.messages, (label, domain))

# integers are formatted by hand, as `print` allocates a string for each
function append_decimal!(buffer::Vector{UInt8}, x::Unsigned)
    first = length(buffer) + 1
    while true
        push!(buffer, UInt8('0') + (x % 10) % UInt8)
        x ÷= 10
        x == 0 && break
    end
    reverse!(buffer, first, length(buffer))
    return buffer
end
append_decimal!(buffer::Vector{UInt8}, x::Integer) = append_decimal!(buffer, unsigned(x))

function nvtxt_record(tracer::NVTXTTracer, kind::String, times, thread::Int, message::String)
    @lock tracer.lock begin
        isopen(tracer.io) || return nothing
        buffer = empty!(tracer.record)
        append!(buffer, codeunits(kind))
        for time in times
            append!(buffer, codeunits(", "))
            append_decimal!(buffer, time)
        end
        append!(buffer, codeunits(", "))
        append_decimal!(buffer, thread)
        append!(buffer, codeunits(", \""))
        append!(buffer, codeunits(message))
        append!(buffer, codeunits("\"\n"))
        write(tracer.io, buffer)
    end
    return nothing
end

trace_range_start(tracer::NVTXTTracer, label, domain) =
    NVTXTRange(time_ns(), Threads.threadid(), nvtxt_message(tracer, label, domain))

function trace_range_end(tracer::NVTXTTracer, range::NVTXTRange)
    stop = time_ns()
    return nvtxt_record(tracer, "RangeStartEnd", (range.start, stop), range.thread, range.message)
end

function trace_mark(tracer::NVTXTTracer, label, domain)
    time = time_ns()
    return nvtxt_record(tracer, "Marker", (time,), Threads.threadid(), nvtxt_message(tracer, label, domain))
end

function nvtxt_path(setting::AbstractString)
    path = setting in ("1", "true", "yes") ? "ka-%p.nvtxt" : setting
    return replace(path, "%p" => string(getpid()))
end

function init_profiling()
    setting = Base.get(ENV, "JULIA_KA_NVTXT", "")
    if !isempty(setting) && !(setting in ("0", "false", "no"))
        tracer = register_tracer!(NVTXTTracer(nvtxt_path(setting)))
        atexit() do
            unregister_tracer!(tracer)
            close(tracer)
        end
    end
    return
end
