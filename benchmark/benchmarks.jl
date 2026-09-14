# To run
# using KernelAbstractions, PkgBenchmark
# result = benchmarkpkg(KernelAbstractions, BenchmarkConfig(env=Dict("KA_BACKEND"=>"CPU", "JULIA_NUM_THREADS"=>"auto")))
# export_markdown("perf.md", result)

using BenchmarkTools
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

# `@Const` asserts that an argument is not written by the kernel. Here that lets the
# compiler hoist the loads of `a`, `b` and `c` out of the loop; without it the store to
# `out` may alias them, so all three have to be reloaded on every step. The two kernels
# below are identical apart from the annotations, so the difference between them is what
# `@Const` buys on a given backend.
@kernel function relax_kernel!(out, @Const(a), @Const(b), @Const(c), ::Val{steps}) where {steps}
    I = @index(Global)
    for _ in 1:steps
        @inbounds out[I] = a[I] * out[I] + b[I] * c[I]
    end
end

@kernel function relax_kernel_unmarked!(out, a, b, c, ::Val{steps}) where {steps}
    I = @index(Global)
    for _ in 1:steps
        @inbounds out[I] = a[I] * out[I] + b[I] * c[I]
    end
end

SUITE["const"] = BenchmarkGroup()

# The two variants share their inputs and are measured on warm data, rather than
# allocating in a setup, so that the comparison isn't dominated by memory traffic.
let steps = Val(16), variants = ("@Const" => relax_kernel!, "unmarked" => relax_kernel_unmarked!)
    for (name, _) in variants
        variant = BenchmarkGroup()
        for T in (Float32, Float64)
            variant["$T"] = BenchmarkGroup()
        end
        SUITE["const"][name] = variant
    end
    for T in (Float32, Float64), N in (65536, 262144)
        a = rand!(KernelAbstractions.zeros(BACKEND, T, N))
        b = rand!(KernelAbstractions.zeros(BACKEND, T, N))
        c = rand!(KernelAbstractions.zeros(BACKEND, T, N))
        for (name, kernel_fun) in variants
            kernel = kernel_fun(BACKEND, 1024)
            out = rand!(KernelAbstractions.zeros(BACKEND, T, N))
            SUITE["const"][name]["$T"][N] = @benchmarkable begin
                $kernel($out, $a, $b, $c, $steps, ndrange = $N)
                KernelAbstractions.synchronize($BACKEND)
            end
        end
    end
end

# Launch overhead: a problem of a single workgroup, so that the time is dominated by the
# host-side work of a launch. The kernel is constructed in the setup and the backend
# synchronized in the teardown, so only the launch itself is measured.
@kernel function scale_kernel!(A, @Const(B))
    I = @index(Global)
    @inbounds A[I] = 2 * B[I]
end

@kernel function scale_kernel_3d!(A, @Const(B))
    i, j, k = @index(Global, NTuple)
    @inbounds A[i, j, k] = 2 * B[i, j, k]
end

SUITE["launch"] = BenchmarkGroup()

let launch = BenchmarkGroup(), n = 16, dims = (4, 4, 4)
    launch["dynamic workgroup, dynamic ndrange"] = @benchmarkable kernel(A, B, ndrange = $n) setup = (
        kernel = scale_kernel!($BACKEND);
        A = KernelAbstractions.zeros($BACKEND, Float64, $n);
        B = rand!(KernelAbstractions.zeros($BACKEND, Float64, $n))
    ) teardown = KernelAbstractions.synchronize($BACKEND)

    launch["dynamic workgroup, dynamic ndrange, workgroupsize given"] = @benchmarkable kernel(A, B, ndrange = $n, workgroupsize = $n) setup = (
        kernel = scale_kernel!($BACKEND);
        A = KernelAbstractions.zeros($BACKEND, Float64, $n);
        B = rand!(KernelAbstractions.zeros($BACKEND, Float64, $n))
    ) teardown = KernelAbstractions.synchronize($BACKEND)

    launch["static workgroup, dynamic ndrange"] = @benchmarkable kernel(A, B, ndrange = $n) setup = (
        kernel = scale_kernel!($BACKEND, $n);
        A = KernelAbstractions.zeros($BACKEND, Float64, $n);
        B = rand!(KernelAbstractions.zeros($BACKEND, Float64, $n))
    ) teardown = KernelAbstractions.synchronize($BACKEND)

    launch["static workgroup, static ndrange"] = @benchmarkable kernel(A, B) setup = (
        kernel = scale_kernel!($BACKEND, $n, $n);
        A = KernelAbstractions.zeros($BACKEND, Float64, $n);
        B = rand!(KernelAbstractions.zeros($BACKEND, Float64, $n))
    ) teardown = KernelAbstractions.synchronize($BACKEND)

    launch["3D static workgroup, dynamic ndrange"] = @benchmarkable kernel(A, B, ndrange = $dims) setup = (
        kernel = scale_kernel_3d!($BACKEND, $dims);
        A = KernelAbstractions.zeros($BACKEND, Float64, $dims...);
        B = rand!(KernelAbstractions.zeros($BACKEND, Float64, $dims...))
    ) teardown = KernelAbstractions.synchronize($BACKEND)

    launch["3D static workgroup, static ndrange"] = @benchmarkable kernel(A, B) setup = (
        kernel = scale_kernel_3d!($BACKEND, $dims, $dims);
        A = KernelAbstractions.zeros($BACKEND, Float64, $dims...);
        B = rand!(KernelAbstractions.zeros($BACKEND, Float64, $dims...))
    ) teardown = KernelAbstractions.synchronize($BACKEND)

    SUITE["launch"] = launch
end

# Host-side partitioning of the iteration space, independent of the backend.
let partition = BenchmarkGroup(), ndrange = Ref((1024,)), workgroupsize = Ref((16,))
    partition["dynamic workgroup, dynamic ndrange"] = @benchmarkable KernelAbstractions.partition(kernel, $ndrange[], $workgroupsize[]) setup = (
        kernel = scale_kernel!($BACKEND)
    )
    partition["static workgroup, dynamic ndrange"] = @benchmarkable KernelAbstractions.partition(kernel, $ndrange[], nothing) setup = (
        kernel = scale_kernel!($BACKEND, 16)
    )
    partition["static workgroup, static ndrange"] = @benchmarkable KernelAbstractions.partition(kernel, nothing, nothing) setup = (
        kernel = scale_kernel!($BACKEND, 16, 1024)
    )
    SUITE["partition"] = partition
end
