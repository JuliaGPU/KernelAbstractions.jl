# EXCLUDE FROM TESTING
using KernelAbstractions
using CUDA

if has_cuda_gpu()
    CUDA.allowscalar(false)
else
    exit()
end

using MPI

backend(A) = typeof(A) <: Array ? CPU() : CUDABackend()

function mpiyield()
    MPI.Iprobe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, MPI.COMM_WORLD)
    yield()
end

function __Irecv!(request, buf, src, tag, comm)
    request[1] = MPI.Irecv!(buf, src, tag, comm)
end

function __Isend!(request, buf, dst, tag, comm)
    request[1] = MPI.Isend(buf, dst, tag, comm)
end

function __testall!(requests)
    done = false
    while !done
        done, _ = MPI.Testall!(requests)
        yield()
    end
end

function exchange!(h_send_buf, d_recv_buf, h_recv_buf, src_rank, dst_rank,
                   send_request, recv_request, comm)

    __Irecv!(recv_request, h_recv_buf, src_rank, 666, comm)
    recv = Base.Threads.@spawn begin 
        __testall!(recv_request)
        async_copy!(backend(d_recv_buf), d_recv_buf, h_recv_buf; progress=mpiyield)
        KernelAbstractions.synchronize(backend(d_recv_buf))
    end

    send = Base.Threads.@spawn begin
        __Isend!(send_request, h_send_buf, dst_rank, 666, comm)
        __testall!(send_request)
    end

    return recv, send
end

function main()
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    MPI.Barrier(comm)

    dst_rank = mod(MPI.Comm_rank(comm)+1, MPI.Comm_size(comm))
    src_rank = mod(MPI.Comm_rank(comm)-1, MPI.Comm_size(comm))

    T = Int64
    M = 10

    d_recv_buf = CUDA.zeros(T, M)
    h_send_buf = zeros(T, M)
    h_recv_buf = zeros(T, M)

    fill!(d_recv_buf, -1)
    fill!(h_send_buf, MPI.Comm_rank(comm))
    fill!(h_recv_buf, -1)

    send_request = fill(MPI.REQUEST_NULL, 1)
    recv_request = fill(MPI.REQUEST_NULL, 1)

    KernelAbstractions.synchronize(backend(d_recv_buf))

    recv_task, send_task = exchange!(h_send_buf, d_recv_buf, h_recv_buf,
                                       src_rank, dst_rank, send_request,
                                       recv_request, comm)
    wait(recv_task)
    wait(send_task)

    @test all(d_recv_buf .== src_rank)
end

main()
