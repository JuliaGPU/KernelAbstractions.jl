# To run
# using KernelAbstractions, PkgBenchmark
# result = benchmarkpkg(KernelAbstractions, BenchmarkConfig(env=Dict("KA_BACKEND"=>"CPU", "JULIA_NUM_THREADS"=>"auto")))
# export_markdown("perf.md", result)

using ValgrindBenchmarkTools
using KernelAbstractions
using Random

if !haskey(ENV, "KA_BACKEND")
    const BACKEND = CPU()
else
    backend = ENV["KA_BACKEND"]
    if backend == "CPU"
        const BACKEND = CPU()
    elseif backend == "CUDA"
        using CUDA
        const BACKEND = CUDABackend()
    else
        error("Backend $backend not recognized")
    end
end

const SUITE = BenchmarkGroup()

@kernel function saxpy_kernel!(Z, a, @Const(X), @Const(Y))
    I = @index(Global)
    @inbounds Z[I] = a * X[I] + Y[I]
end

SUITE["saxpy"] = BenchmarkGroup()

let static = BenchmarkGroup()
    for T in (Float16, Float32, Float64)
        dtype = BenchmarkGroup()
        for N in (64, 256, 512, 1024, 2048, 4096, 16384, 32768, 65536, 262144, 1048576)
            dtype[N] = @benchmarkable begin
                kernel = saxpy_kernel!($BACKEND, 1024)
                kernel(Z, convert($T, 2.0), X, Y, ndrange = size(Z))
            end setup = (
                X = rand!(KernelAbstractions.zeros($BACKEND, $T, $N));
                Y = rand!(KernelAbstractions.zeros($BACKEND, $T, $N));
                Z = KernelAbstractions.zeros($BACKEND, $T, $N)
            )
        end
        static["$T"] = dtype
    end
    SUITE["saxpy"]["static workgroup=(1024,)"] = static
end

let default = BenchmarkGroup()
    for T in (Float16, Float32, Float64)
        dtype = BenchmarkGroup()
        for N in (64, 256, 512, 1024, 2048, 4096, 16384, 32768, 65536, 262144, 1048576)
            dtype[N] = @benchmarkable begin
                kernel = saxpy_kernel!($BACKEND)
                kernel(Z, convert($T, 2.0), X, Y, ndrange = size(Z))
            end setup = (
                X = rand!(KernelAbstractions.zeros($BACKEND, $T, $N));
                Y = rand!(KernelAbstractions.zeros($BACKEND, $T, $N));
                Z = KernelAbstractions.zeros($BACKEND, $T, $N)
            )
        end
        default["$T"] = dtype
    end
    SUITE["saxpy"]["default"] = default
end
