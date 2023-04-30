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
function measure_membw(backend = CPU(); verbose = true, N = 1024 * 500_000, dtype = Float32,
                       init = :parallel)
    bytes = 3 * sizeof(dtype) * N # num bytes transferred in SAXPY
    flops = 2 * N # num flops in SAXY

    a = dtype(3.1415)
    X = allocate(backend, dtype, N)
    Y = allocate(backend, dtype, N)
    if init == :parallel && backend isa CPU
        Threads.@threads :static for i in eachindex(Y)
            X[i] = rand()
            Y[i] = rand()
        end
    else
        rand!(X)
        rand!(Y)
    end
    workgroup_size = 1024

    t = @belapsed begin
        kernel = saxpy_kernel($backend, $workgroup_size, $(size(Y)))
        kernel($a, $X, $Y, ndrange = $(size(Y)))
        KernelAbstractions.synchronize($backend)
    end evals=2 samples=10

    mem_rate = bytes * 1e-9 / t # GB/s
    flop_rate = flops * 1e-9 / t # GFLOP/s

    if verbose
        println("\tMemory Bandwidth (GB/s): ", round(mem_rate; digits = 2))
        println("\tCompute (GFLOP/s): ", round(flop_rate; digits = 2))
    end
    return mem_rate, flop_rate
end

measure_membw(CPU());

measure_membw(CPU(; static=true));
