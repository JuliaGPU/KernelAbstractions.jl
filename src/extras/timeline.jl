module Timeline

using Requires
export @range, mark

module NVTXT
    const LOG_FILE=Ref{IOStream}()
    const SHOULD_LOG=Ref{Bool}(false)

    function __init__()
        if haskey(ENV, "KERNELABSTRACTIONS_TIMELINE")
            SHOULD_LOG[] = true
        else
            SHOULD_LOG[] = false
            return
        end
        pid = Libc.getpid()
        LOG_FILE[] = open("ka-$pid.nvtxt", "w")
        initialize()
        atexit() do
            close(LOG_FILE[])
        end
    end

    function initialize()
        SHOULD_LOG[] || return
        io = LOG_FILE[]
        pid = Libc.getpid()
        print(io, """
        SetFileDisplayName, KernelAbstractions
        @RangeStartEnd, Start, End, ThreadId, Message
        ProcessId = $pid
        CategoryId = 1
        Color = Blue
        TimeBase = Manual
        @RangePush, Time, ThreadId, Message
        ProcessId = $pid
        CategoryId = 1
        Color = Blue
        TimeBase = Manual
        @RangePop, Time, ThreadId
        ProcessId = $pid
        TimeBase = Manual
        @Marker, Time, ThreadId, Message
        ProcessId = $pid
        CategoryId = 1
        Color = Blue
        TimeBase = Manual
        """)
    end

    function push_range(msg)
        SHOULD_LOG[] || return
        time = time_ns()
        io = LOG_FILE[]
        print(io, "RangePush, ")
        print(io, time)
        println(io, ", ", Base.Threads.threadid(), ", \"", msg, "\"")
    end

    function pop_range()
        SHOULD_LOG[] || return
        time = time_ns()
        io = LOG_FILE[]
        print(io, "RangePop, ")
        print(io, time)
        println(io, ", ", Base.Threads.threadid())
    end

    struct Range
        start::UInt64
        msg::String
    end

    start_range(msg::String) = Range(time_ns(), msg)
    function end_range(r::Range)
        SHOULD_LOG[] || return
        time = time_ns()
        io = LOG_FILE[]
        print(io, "RangeStartEnd, ")
        show(io, r.start)
        print(io, ", ")
        show(io, time)
        println(io, ", ", Base.Threads.threadid(), ", \"", r.msg, "\"")
    end

    function mark(msg::String)
        SHOULD_LOG[] || return
        time = time_ns()
        io = LOG_FILE[]
        print(io, "Marker, ")
        show(io, time)
        println(io, ", ", Base.Threads.threadid(), ", \"", msg, "\"")
    end
end # NVTXT

_mark(msg) = NVTXT.mark(msg)
_push_range(msg) = NVTXT.push_range(msg)
_pop_range() = NVTXT.pop_range()
_start_range(msg) = NVTXT.start_range(msg)
_end_range(r) = NVTXT.end_range(r)

@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
    # replace implementations
    import CUDAnative.NVTX

    _mark(msg) = NVTX.mark(msg)
    _push_range(msg) = NVTX.push_range(msg)
    _pop_range() = NVTX.pop_range()
    _start_range(msg) = NVTX.start_range(msg)
    _end_range(r) = NVTX.end_range(r)
end

import Base: invokelatest
mark(msg) = invokelatest(_mark, msg)
push_range(msg) = invokelatest(_push_range, msg)
pop_range() = invokelatest(_pop_range)
start_range(msg) = invokelatest(_start_range, msg)
end_range(r) = invokelatest(_end_range, r)

"""
    @range "msg" ex
Create a new range and execute `ex`. The range is popped automatically afterwards.
See also: [`range`](@ref)
"""
macro range(msg, ex)
    quote
        local range = $start_range($(esc(msg)))
        local ret = $(esc(ex))
        $end_range(range)
        ret
    end
end

end
