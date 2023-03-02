# EXCLUDE FROM TESTING
using KernelAbstractions
using MPI

function mpiprogress()
    MPI.Iprobe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, MPI.COMM_WORLD)
    yield()
end

function test!(req)
    done = false
    while !done
        done, _ = MPI.Test(req, MPI.Status)
        yield()
    end
end

function exchange!(h_send_buf, d_recv_buf, h_recv_buf, src_rank, dst_rank, comm)
    recv_req = MPI.Irecv!(h_recv_buf, src_rank, 666, comm)
    recv = Base.Threads.@spawn begin 
        test!(recv_req)
        KernelAbstractions.copyto!(backend, d_recv_buf, h_recv_buf)
        # Call back into MPI to gurantuee that we can make progress
        KernelAbstractions.synchronize(backend; progress = mpiprogress)
    end

    send = Base.Threads.@spawn begin
        send_req = MPI.Isend!(h_send_buf, dst_rank, 666, comm)
        test!(send_req)
    end

    return recv, send
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

    recv_task, send_task = exchange!(h_send_buf, d_recv_buf, h_recv_buf,
                                       src_rank, dst_rank, comm)
    wait(recv_task)
    wait(send_task)

    @test all(d_recv_buf .== src_rank)
end

main(backend)
