module Timeline

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
        TimeBase = ClockMonotonicRaw
        @RangePush, Time, ThreadId, Message
        ProcessId = $pid
        CategoryId = 1
        Color = Blue
        TimeBase = ClockMonotonicRaw
        @RangePop, Time, ThreadId
        ProcessId = $pid
        TimeBase = ClockMonotonicRaw
        @Marker, Time, ThreadId, Message
        ProcessId = $pid
        CategoryId = 1
        Color = Blue
        TimeBase = ClockMonotonicRaw
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

function range_push end
function mark end

end