# EXCLUDE FROM TESTING
using KernelAbstractions
using MPI

mutable struct Exchanger
    @atomic done::Bool
    top::Base.Event
    bottom::Base.Event
    @atomic err
    task::Task

    function Exchanger(f::F) where F
        top = Base.Event(#=autoreset=# true)
        bottom = Base.Event(#=autoreset=# true)
        this = new(false, top, bottom, nothing)

        this.task = Threads.@spawn begin
            try
                while !(@atomic this.done)
                    wait(top)
                    f()
                    notify(bottom)
                end
            catch err
                @atomic this.done = true
                @atomic this.err = err
            end
        end
        return this
    end
end

Base.isdone(exc::Exchanger) = @atomic exc.done
function Base.notify!(exc::Exchanger)
    if !(@atomic exc.done)
        notify!(exc.top)
    else
        error("Exchanger is not running")
    end
end
function Base.wait(exc::Exchanger)
    if !(@atomic exc.done)
        wait(exc.top)
    else
        error("Exchanger is not running")
    end
end




# TODO: Implement in MPI.jl
function cooperative_test!(req)
    done = false
    while !done
        done, _ = MPI.Test(req, MPI.Status)
        yield()
    end
end

function cooperative_wait(task::Task)
    while !Base.istaskdone(task)
        MPI.Iprobe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, MPI.COMM_WORLD)
        yield()
    end
    wait(task)
end

function cooperative_wait(task::Base.Event)
    while !(@atomic task.set)
        MPI.Iprobe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, MPI.COMM_WORLD)
        yield()
    end
    wait(task)
end

function setup_exchange!(h_send_buf, d_recv_buf, h_recv_buf, src_rank, dst_rank, comm)
    recv_barrier = Base.Event(true)
    recv = Base.Threads.@spawn begin
        KernelAbstractions.priority!(backend, :high)
        while yield()::Bool
            recv_req = MPI.Irecv!(h_recv_buf, src_rank, 666, comm)
            cooperative_test!(recv_req)
            KernelAbstractions.copyto!(backend, d_recv_buf, h_recv_buf)
            KernelAbstractions.synchronize(backend) # Gurantueed to be cooperative
            notify(recv_barrier)
        end
    end
    errormonitor(recv)

    send_barrier = Base.Event(true)
    send = Base.Threads.@spawn begin
        while yield()::Bool
            send_req = MPI.Isend!(h_send_buf, dst_rank, 666, comm)
            cooperative_test!(send_req)
            notify(send_barrier)
        end
    end
    errormonitor(send)

    return (recv, recv_barrier), (send, send_barrier)
end

function main(backend)
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    MPI.Barrier(comm)

    dst_rank = mod(MPI.Comm_rank(comm)+1, MPI.Comm_size(comm))
    src_rank = mod(MPI.Comm_rank(comm)-1, MPI.Comm_size(comm))

    T = Int64
    M = 10

    d_recv_buf = allocate(backend, T, M)
    fill!(d_recv_buf, -1)

    h_send_buf = zeros(T, M)
    h_recv_buf = zeros(T, M)
    fill!(h_send_buf, MPI.Comm_rank(comm))
    fill!(h_recv_buf, -1)

    KernelAbstractions.synchronize(backend)

    (recv, recv_barrier), (send, send_barrier) = setup_exchange!(h_send_buf, d_recv_buf, h_recv_buf,
                                                                 src_rank, dst_rank, comm)

    for i in 1:10
        yieldto(recv, true) # trigger recv task
        yieldto(send, true) # trigger recv task

        # do something useful

        cooperative_wait(recv_barrier)
        cooperative_wait(send_barrier)

        @test all(d_recv_buf .== src_rank)
        d_recv_buf .= 0
    end

    yieldto(recv, false) # optional
    yieldto(send, false) # optional
end

main(backend)
