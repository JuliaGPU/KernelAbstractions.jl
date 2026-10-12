# EXCLUDE FROM TESTING
# Run with `POCL_AFFINITY=1` to pin the threads that run the kernels, e.g.
#   POCL_AFFINITY=1 julia -t 128 examples/numa_aware.jl
using BenchmarkTools
using KernelAbstractions

@kernel function saxpy_kernel(a, @Const(X), Y)
    I = @index(Global)
    @inbounds Y[I] = a * X[I] + Y[I]
end

@kernel function fill_kernel(A, x)
    I = @index(Global)
    @inbounds A[I] = x
end

"""
  measure_membw(; kwargs...) -> membw, flops

Estimate the memory bandwidth (GB/s) by performing a time measurement of a
SAXPY kernel. Returns the memory bandwidth (GB/s) and the compute (GFLOP/s).
"""
function measure_membw(
        backend = CPU(); verbose = true, N = 1024 * 500_000, dtype = Float32,
        init = :parallel,
    )
    bytes = 3 * sizeof(dtype) * N # num bytes transferred in SAXPY
    flops = 2 * N # num flops in SAXY
    workgroup_size = 1024

    a = dtype(3.1415)
    X = KernelAbstractions.allocate(backend, dtype, N)
    Y = KernelAbstractions.allocate(backend, dtype, N)
    if init == :serial
        # The calling thread touches all the memory first
        fill!(X, dtype(1))
        fill!(Y, dtype(2))
    else
        # The threads that run the kernels touch the memory first
        fill_kernel(backend, workgroup_size)(X, dtype(1), ndrange = size(X))
        fill_kernel(backend, workgroup_size)(Y, dtype(2), ndrange = size(Y))
        KernelAbstractions.synchronize(backend)
    end

    t = @belapsed begin
        kernel = saxpy_kernel($backend, $workgroup_size, $(size(Y)))
        kernel($a, $X, $Y, ndrange = $(size(Y)))
        KernelAbstractions.synchronize($backend)
    end evals = 2 samples = 10

    mem_rate = bytes * 1.0e-9 / t # GB/s
    flop_rate = flops * 1.0e-9 / t # GFLOP/s

    if verbose
        println("\tMemory Bandwidth (GB/s): ", round(mem_rate; digits = 2))
        println("\tCompute (GFLOP/s): ", round(flop_rate; digits = 2))
    end
    return mem_rate, flop_rate
end

measure_membw(CPU());

# On a system with multiple NUMA domains, this places all the memory in the domain of the calling thread
# measure_membw(CPU(); init=:serial);
