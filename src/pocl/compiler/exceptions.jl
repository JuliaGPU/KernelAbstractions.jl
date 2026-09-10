# support for device-side exceptions

## exception type

"""
    KernelException

An exception thrown during kernel execution on `dev`, detected when synchronizing
(`KernelAbstractions.synchronize`, or the implicit synchronization at the end of a launch).

How much is known about the exception depends on the debug level the kernel was compiled
with (the session's `-g` level, or `@opencl debug_level=`):

- `0`: only that an exception was thrown;
- `1`: additionally its type `name` and `reason`, for exceptions thrown by Julia's runtime
  (bounds errors, domain errors, ...);
- `2`: additionally the position of the faulting work-item (`work_item` is its local id,
  `work_group` the id of its work-group), the name of any other exception, and a device-side
  `backtrace` as `(function, file, line)` tuples.

`dev` identifies the device whose mailbox reported the exception.
Fields that were not recorded are empty strings, empty vectors, or all-zero tuples.
"""
struct KernelException <: Exception
    dev::cl.Device
    name::String
    reason::String
    work_item::NTuple{3, Int}
    work_group::NTuple{3, Int}
    backtrace::Vector{Tuple{String, String, Int}}   # (function, file, line) per frame
end

# Positions are 1-based, so zero coordinates indicate that level 2 details were not recorded.
function Base.showerror(io::IO, err::KernelException)
    name = isempty(err.name) ? "exception" : err.name
    article = first(uppercase(name)) in ('A', 'E', 'I', 'O', 'U') ? "An" : "A"
    print(io, "KernelException: $article $name was thrown")
    if err.work_item != (0, 0, 0)
        work_item = join(err.work_item, '×')
        work_group = join(err.work_group, '×')
        print(io, " by work-item $work_item in work-group $work_group")
    end
    print(io, " on device ", err.dev.name)
    isempty(err.reason) || print(io, ": ", err.reason)
    if err.work_item == (0, 0, 0)
        print(io, "\nFor more details, run Julia with `-g2`")
    else
        print(io, "\nStacktrace:")
        for (i, (func, file, line)) in enumerate(err.backtrace)
            print(io, "\n [", i, "] ", func)
            isempty(file) || print(io, " at ", file, ":", line)
        end
    end
    return
end

# decode a null-terminated mailbox text buffer into a `String`
function exception_string(bytes::NTuple{N, UInt8}) where {N}
    len = something(findfirst(iszero, bytes), N + 1) - 1
    return String(UInt8[bytes[i] for i in 1:len])
end


## exception mailbox

# One mailbox per (context, device), shared by its queues. Track submissions so a host read
# waits for every possible writer.
mutable struct ExceptionMailbox
    # the device this mailbox reports for
    const dev::cl.Device
    # Host memory the kernel writes through the address in its `KernelState`. pocl's CPU
    # device shares the host address space, so a plain allocation is reachable from device
    # code. Never freed: a kernel that outlives the host's interest still holds the address.
    const ptr::Ptr{ExceptionInfo_st}
    # serialize launches and checks for this mailbox
    const lock::ReentrantLock
    # queues that have launched kernels since the mailbox was last checked
    const pending::Set{cl.CmdQueue}
    # assigned under the lock; distinguishes work-items from different launches
    launch_id::UInt64
end

function ExceptionMailbox(dev::cl.Device)
    ptr = convert(Ptr{ExceptionInfo_st}, Libc.malloc(sizeof(ExceptionInfo_st)))
    ptr == C_NULL && throw(OutOfMemoryError())
    unsafe_store!(ptr, ExceptionInfo_st())
    return ExceptionMailbox(dev, ptr, ReentrantLock(), Set{cl.CmdQueue}(), UInt64(0))
end

# Device-scoped atomics cannot protect a mailbox shared by different devices, so key the
# mailboxes by device as well as context. These strong references retain their contexts.
const exception_mailboxes = Dict{Tuple{cl.Context, cl.Device}, ExceptionMailbox}()
const exception_mailboxes_lock = ReentrantLock()

function exception_mailbox(ctx::cl.Context = context(), dev::cl.Device = device())
    return Base.@lock exception_mailboxes_lock begin
        get!(() -> ExceptionMailbox(dev), exception_mailboxes, (ctx, dev))
    end
end

# Launch while holding the mailbox lock, so a concurrent check cannot finish the queue
# between the enqueue and recording it as pending. Keep this in an ordinary function: the
# generated `AbstractKernel` call cannot contain a closure or `do` block on Julia 1.13.
function launch_with_exception_mailbox(
        kernel::cl.Kernel, call_tt::Type, random_seed::UInt32, args...;
        kwargs...
    )
    mailbox = exception_mailbox()
    return Base.@lock mailbox.lock begin
        mailbox.launch_id += UInt64(1)
        state = KernelState(random_seed, UInt64(UInt(mailbox.ptr)), mailbox.launch_id)
        event = cl.clcall(kernel, call_tt, state, args...; kwargs...)
        push!(mailbox.pending, queue())
        event
    end
end

# read out and reset the mailbox; the caller must have completed every queue that could
# still be writing to it
function take_exception!(mailbox::ExceptionMailbox)
    ptr = mailbox.ptr
    unsafe_load(convert(Ptr{Int32}, ptr)) == 0 && return nothing
    info = unsafe_load(ptr)
    nframes = min(Int(info.num_frames), EXCEPTION_MAX_FRAMES)
    backtrace = Tuple{String, String, Int}[
        (
            exception_string(info.frames[i].func),
            exception_string(info.frames[i].file),
            Int(info.frames[i].line),
        ) for i in 1:nframes
    ]
    # clear the flag, lock and payload so the mailbox is reusable (and a later check
    # doesn't re-report the same exception)
    unsafe_store!(ptr, ExceptionInfo_st())
    return KernelException(
        mailbox.dev, exception_string(info.name), exception_string(info.reason),
        Int.(info.work_item[1:3]), Int.(info.work_group[1:3]), backtrace
    )
end

"""
    check_exceptions()

Synchronize every queue that has launched a kernel on the current device since the last
check, then, if a kernel threw, rethrow it host-side as a [`KernelException`](@ref).

The exception mailbox is shared by the queues targeting the same device in a context, so
this may wait for and surface an exception from another queue on that device.
"""
function check_exceptions()
    mailbox = Base.@lock exception_mailboxes_lock begin
        # `context()` and `device()` initialize the task's POCL state; don't pay for that
        # (or force it on a task that never launched) just to find an empty registry
        isempty(exception_mailboxes) ? nothing :
            get(exception_mailboxes, (context(), device()), nothing)
    end
    mailbox === nothing && return
    exc = Base.@lock mailbox.lock begin
        isempty(mailbox.pending) && return
        # The mailbox cannot be read while a kernel may still write to it. Finish every
        # queue that launched with its address; the launch-side lock keeps a new enqueue
        # from appearing between this loop and the reset below.
        for pending in mailbox.pending
            cl.clFinish(pending)
        end
        empty!(mailbox.pending)
        take_exception!(mailbox)
    end
    exc === nothing || throw(exc)
    return
end
