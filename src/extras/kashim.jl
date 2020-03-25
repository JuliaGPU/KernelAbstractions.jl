module KAShim

import KAShim_jll: libkashim

function malloc()
    ccall((:ka_barrier_malloc, libkashim), Ptr{Cvoid}, ())
end

function free(ptr)
    ccall((:ka_barrier_free, libkashim), Cvoid, (Ptr{Cvoid},), ptr)
end

function init(ptr, n)
    err = ccall((:ka_barrier_init, libkashim), Cint, (Ptr{Cvoid},Cuint), ptr, n)
    if err < 0
        error("Couldn't initialize barrier")
    end
    return nothing
end

function destroy(ptr)
    ccall((:ka_barrier_destroy, libkashim), Cvoid, (Ptr{Cvoid},), ptr)
end

function __wait(ptr)
    # this could be a `@threadcall`
    ccall((:ka_barrier_wait, libkashim), Cint, (Ptr{Cvoid},), ptr)
end

struct Payload
    handle::Ptr{Cvoid}
    barrier::Ptr{Cvoid}
    function Payload(handle::Ptr{Cvoid})
        barrier = malloc()
        init(barrier, 2)
        new(handle, barrier)
    end
end

function waiter(event)
    cond = Base.AsyncCondition()
    payload = Ref(Payload(cond.handle))

    # Task that is used to perform the waiting on cpu_event and the subsequent
    # blocking barrier. We perhaps could use `@threadcall` to use LibUV threads
    # to do the barrier_wait, but that barrier_wait should be cheap since the 
    # callback thread has already reached it.
    task = Threads.@spawn begin
        GC.@preserve cond payload begin
            # wait on AsyncCondition, e.g. until CUDA notified use
            Base.wait(cond)
            try
                # wait on actual work
                Base.wait(event)
            catch err
                bt = catch_backtrace()
                @error "Error thrown during wait on event" _ex=(err, bt)
            finally
                # now notify callback, this uses a `uv_barrier_t` under the hood
                # and so will block until both sides made progress
                barrier = payload[].barrier
                if __wait(barrier) > 0
                    destroy(barrier)
                    free(barrier)
                end
            end
        end
        return nothing
    end
    return payload, task
end


function callback_ptr()
    cglobal((:ka_callback, libkashim))
end

end