using KernelAbstractions, MPI, CuArrays, CUDAnative, CUDAdrv

@kernel function kernel!(c, @Const(a), @Const(b))
  i = (blockIdx().x-1) * blockDim().x + threadIdx().x

  @inbounds a_val = a[i]
  @inbounds b_val = b[i]
  @inbounds c_val = c[i]

  for j in 1:99
    a_val = a_val * b_val + c_val
    b_val = a_val * b_val + c_val
    c_val = a_val * b_val + c_val
  end

  @inbounds c[i] = a_val * b_val + c_val

  return
end

function main()
  if !MPI.Initialized()
    MPI.Init()
  end
  comm = MPI.COMM_WORLD
  local_comm = MPI.Comm_split_type(comm, MPI.MPI_COMM_TYPE_SHARED,
                                   MPI.Comm_rank(comm))
  device!(MPI.Comm_rank(local_comm) % length(devices()))
  copystream = CuStream(CUDAdrv.STREAM_NON_BLOCKING)
  CuArrays.allowscalar(false)

  synchronize(copystream)
  MPI.Barrier(comm)

  dst_rank = mod(MPI.Comm_rank(comm)+1, MPI.Comm_size(comm))
  src_rank = mod(MPI.Comm_rank(comm)-1, MPI.Comm_size(comm))

  T = Float32
  M = 1024
  N = M*100

  A = CuArray(rand(T, M, N))
  B = CuArray(rand(T, M, N))
  C = CuArray(rand(T, M, N))
  d = CuArray(rand(T, M))

  b = Array{T}(undef, M)
  c = Array{T}(undef, M)
  pin!(b)
  pin!(c)

  len = length(A)
  threads = min(len, 1024)
  blocks = div(len, threads)

  compevent = recordevent(copystream)
  copyevent = recordevent(copystream)
  for j = 1:100
    copyevent = async_copy!(pointer(b), pointer(B), M, stream=copystream, dependencies=compevent)
    compevent = kernel!(CUDA(), 256)(C, A, B, ndrange=length(C), dependencies=compevent)

    wait(copyevent)
    rreq = MPI.Irecv!(c, src_rank, 222, comm)
    sreq = MPI.Isend(b, dst_rank, 222, comm)

    stats = MPI.Waitall!([sreq, rreq])
    copyevent = async_copy!(pointer(d), pointer(c), M, stream=copystream)
    copyevent = async_copy!(pointer(C), pointer(d), M, stream=copystream, dependencies=copyevent)
    compevent = kernel!(CUDA(), 256)(C, A, B, ndrange=length(C), dependencies=copyevent)
  end
end

main()
