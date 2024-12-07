# EXCLUDE FROM TESTING
using BenchmarkTools
using Statistics
using Random
using ThreadPinning
using KernelAbstractions

ThreadPinning.pinthreads(:numa)

@kernel function saxpy_kernel(a, @Const(X), Y)
    I = @index(Global)
    @inbounds Y[I] = a * X[I] + Y[I]
end

"""
  measure_membw(; kwargs...) -> membw, flops

Estimate the memory bandwidth (GB/s) by performing a time measurement of a
SAXPY kernel. Returns the memory bandwidth (GB/s) and the compute (GFLOP/s).
"""
function measure_membw(
        backend = OpenCLBackend(); verbose = true, N = 1024 * 500_000, dtype = Float32,
        init = :parallel,
    )
    bytes = 3 * sizeof(dtype) * N # num bytes transferred in SAXPY
    flops = 2 * N # num flops in SAXY

    a = dtype(3.1415)
    if init == :serial
        X = rand!(zeros(dtype, N))
        Y = rand!(zeros(dtype, N))
    else
        X = rand!(KernelAbstractions.zeros(backend, dtype, N))
        Y = rand!(KernelAbstractions.zeros(backend, dtype, N))
    end
    workgroup_size = 1024

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

# Static should be much better (on a system with multiple NUMA domains)
measure_membw(OpenCLBackend());
# measure_membw(OpenCLBackend(; static = true));

# The following has significantly worse performance (even on systems with a single memory domain)!
# measure_membw(CPU(); init=:serial);
# measure_membw(CPU(; static=true); init=:serial);
