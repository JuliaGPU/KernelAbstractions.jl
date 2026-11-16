# Compiled kernels, each isolating one feature of KernelAbstractions or of the code it
# generates. Every benchmark launches a kernel that has been compiled for exactly this
# configuration, on data that has been initialized, and waits for it to complete. The
# `performance/julia/...` benchmarks are serial Julia loops doing the same work as a
# kernel, to compare the code generated for a kernel with that of plain Julia.

using KernelAbstractions: @atomic

# Data on the backend, initialized with deterministic values in `[lo, hi)`
function uniform(T, dims...; lo = 0, hi = 1)
    rng = Random.Xoshiro(42)
    host = T.(lo .+ (hi - lo) .* rand(rng, Float64, dims...))
    return copyto!(KernelAbstractions.allocate(BACKEND, T, dims...), host)
end

# A benchmark of a kernel that has been compiled and run once (and checked by `check`)
function measure_kernel(f, check; kind = KERNEL, setup = Returns(nothing))
    setup()
    f()
    check() || error("wrong result")
    return measure(f, kind; setup)
end


## streaming: bandwidth of the cache (8192 elements) and of memory (2^24 elements)

@kernel function triad!(z, a, @Const(x), @Const(y))
    i = @index(Global, Linear)
    @inbounds z[i] = a * x[i] + y[i]
end

const STREAM_SIZES = ("cache" => 8192, "dram" => 2^24)

for (regime, n) in STREAM_SIZES
    bench("performance/pocl/parallel/stream/copy/$regime") do
        x = uniform(Float32, n)
        y = KernelAbstractions.zeros(BACKEND, Float32, n)
        f() = (KernelAbstractions.copyto!(BACKEND, y, x); synchronize(BACKEND))
        measure_kernel(f, () -> y == x)
    end
    for mode in (regime == "dram" ? ("parallel", "serial") : ("parallel",))
        bench("performance/pocl/$mode/stream/triad/$regime") do
            x, y = uniform(Float32, n), uniform(Float32, n)
            z = KernelAbstractions.zeros(BACKEND, Float32, n)
            kernel = triad!(BACKEND, 256)
            f() = (kernel(z, 2.0f0, x, y; ndrange = n); synchronize(BACKEND))
            measure_kernel(f, () -> z == 2.0f0 .* x .+ y)
        end
    end
end


## indexing: the same work, with different kinds of indices and dimensions

@kernel function affine_linear!(y, @Const(x))
    i = @index(Global, Linear)
    @inbounds y[i] = 2.0f0 * x[i] + 1.0f0
end

@kernel function affine_cartesian!(y, @Const(x))
    I = @index(Global, Cartesian)
    @inbounds y[I] = 2.0f0 * x[I] + 1.0f0
end

@kernel function affine_ntuple!(y, @Const(x))
    I = @index(Global, NTuple)
    @inbounds y[I...] = 2.0f0 * x[I...] + 1.0f0
end

for (name, kernelf, dims, groupsize) in (
        ("linear3d", affine_linear!, (128, 128, 64), (8, 8, 4)),
        ("cartesian3d", affine_cartesian!, (128, 128, 64), (8, 8, 4)),
        ("ntuple3d", affine_ntuple!, (128, 128, 64), (8, 8, 4)),
        ("ntuple4d", affine_ntuple!, (32, 32, 32, 32), (4, 4, 4, 4)),
    )
    bench("performance/pocl/parallel/indexing/$name") do
        x = uniform(Float32, dims...)
        y = KernelAbstractions.zeros(BACKEND, Float32, dims...)
        kernel = kernelf(BACKEND, groupsize)
        f() = (kernel(y, x; ndrange = dims); synchronize(BACKEND))
        measure_kernel(f, () -> y == 2.0f0 .* x .+ 1.0f0)
    end
end


## narrow workgroups: the first dimension of a workgroup has only 4 work-items, which
## PoCL's loop over the work-items vectorizes only if the index computation needs no
## runtime checks (#845, #868). The last workgroup of the masked launch is partial, so the
## kernel checks the bounds of its work-items; without a partial workgroup, KA would skip
## that check like `unsafe_indices` does.

@kernel inbounds = true function narrow_masked!(y, @Const(x))
    i, e = @index(Global, NTuple)
    y[i, e] = x[i, e] / (1 + x[i, e] * x[i, e])
end

@kernel inbounds = true unsafe_indices = true function narrow_unmasked!(y, @Const(x))
    i, e = @index(Global, NTuple)
    y[i, e] = x[i, e] / (1 + x[i, e] * x[i, e])
end

const NARROW_DIMS = (4, 2^18)

narrow_reference(x) = x ./ (1 .+ x .* x)

for (name, kernelf, dims) in (
        ("masked", narrow_masked!, (4, 2^18 - 1)),
        ("unmasked", narrow_unmasked!, NARROW_DIMS),
    )
    bench("performance/pocl/serial/narrow/$name") do
        x = uniform(Float64, dims...)
        y = similar(x)
        kernel = kernelf(BACKEND, (4, 64))
        f() = (kernel(y, x; ndrange = dims); synchronize(BACKEND))
        measure_kernel(f, () -> y ≈ narrow_reference(x))
    end
end

function narrow_julia!(y, x)
    @inbounds for e in axes(x, 2), i in axes(x, 1)
        y[i, e] = x[i, e] / (1 + x[i, e] * x[i, e])
    end
    return y
end

bench("performance/julia/serial/narrow/masked-reference") do
    x = Array(uniform(Float64, NARROW_DIMS...))
    y = similar(x)
    measure_kernel(() -> narrow_julia!(y, x), () -> y ≈ narrow_reference(x))
end


## scheduling: a small problem with enough work per work-item to be worth spreading over
## the workers, with the workgroup size chosen by KernelAbstractions or given (#859)

@kernel function iterate!(y, @Const(x), ::Val{n}) where {n}
    i = @index(Global, Linear)
    v = @inbounds x[i]
    for _ in 1:n
        v = v * 0.999f0 + 0.001f0
    end
    @inbounds y[i] = v
end

function iterate_reference(x, n)
    return map(x) do v
        for _ in 1:n
            v = v * 0.999f0 + 0.001f0
        end
        v
    end
end

for (name, groupsize) in (("default", nothing), ("explicit", 256))
    bench("performance/pocl/parallel/scheduling/$name") do
        x = uniform(Float32, 4096)
        y = similar(x)
        kernel = groupsize === nothing ? iterate!(BACKEND) : iterate!(BACKEND, groupsize)
        f() = (kernel(y, x, Val(128); ndrange = length(x)); synchronize(BACKEND))
        measure_kernel(f, () -> y ≈ iterate_reference(Array(x), 128))
    end
end


## `@Const`: asserts that an argument is not written by the kernel, which lets the compiler
## keep the values of `a`, `b` and `c` in registers instead of reloading them after every
## store to `out`. The two kernels are identical apart from the annotations.

@kernel function relax_const!(out, @Const(a), @Const(b), @Const(c), ::Val{steps}) where {steps}
    i = @index(Global, Linear)
    for _ in 1:steps
        @inbounds out[i] = a[i] * out[i] + b[i] * c[i]
    end
end

@kernel function relax_unmarked!(out, a, b, c, ::Val{steps}) where {steps}
    i = @index(Global, Linear)
    for _ in 1:steps
        @inbounds out[i] = a[i] * out[i] + b[i] * c[i]
    end
end

for (name, kernelf) in (("marked", relax_const!), ("unmarked", relax_unmarked!))
    bench("performance/pocl/serial/const/$name") do
        n = 8192
        # |a| < 1/2 and |b c| < 1/4 keep `out` bounded
        a, b, c = uniform(Float64, n; hi = 0.5), uniform(Float64, n; hi = 0.5), uniform(Float64, n; hi = 0.5)
        out = KernelAbstractions.zeros(BACKEND, Float64, n)
        kernel = kernelf(BACKEND, 256)
        f() = (kernel(out, a, b, c, Val(64); ndrange = n); synchronize(BACKEND))
        expected = zeros(n)
        for _ in 1:64
            expected .= Array(a) .* expected .+ Array(b) .* Array(c)
        end
        measure_kernel(f, () -> Array(out) ≈ expected; setup = () -> fill!(out, 0))
    end
end


## math: functions of the math library, and floating-point contraction (#860, #865)

@kernel function map_exp!(y, @Const(x))
    i = @index(Global, Linear)
    @inbounds y[i] = exp(x[i])
end

@kernel function map_sin!(y, @Const(x))
    i = @index(Global, Linear)
    @inbounds y[i] = sin(x[i])
end

for (name, kernelf, f, lo, hi) in (
        ("exp-f64", map_exp!, exp, -1, 1),
        ("sin-f64", map_sin!, sin, -π, π),
    )
    bench("performance/pocl/serial/math/$name") do
        x = uniform(Float64, 65536; lo, hi)
        y = similar(x)
        kernel = kernelf(BACKEND, 256)
        g() = (kernel(y, x; ndrange = length(x)); synchronize(BACKEND))
        measure_kernel(g, () -> y ≈ f.(x))
    end
end

function exp_julia!(y, x)
    @inbounds for i in eachindex(y, x)
        y[i] = exp(x[i])
    end
    return y
end

bench("performance/julia/serial/math/exp-f64-reference") do
    x = Array(uniform(Float64, 65536; lo = -1, hi = 1))
    y = similar(x)
    measure_kernel(() -> exp_julia!(y, x), () -> y ≈ exp.(x))
end

# `a * v + b` may be contracted into a fused multiply-add, `muladd(a, v, b)` asks for one.
# With `a = 1/2` and `b = 1/4`, `v` converges to `1/2`.
@kernel function chain_contract!(y, @Const(x), ::Val{n}) where {n}
    i = @index(Global, Linear)
    v = @inbounds x[i]
    for _ in 1:n
        v = 0.5 * v + 0.25
    end
    @inbounds y[i] = v
end

@kernel function chain_muladd!(y, @Const(x), ::Val{n}) where {n}
    i = @index(Global, Linear)
    v = @inbounds x[i]
    for _ in 1:n
        v = muladd(0.5, v, 0.25)
    end
    @inbounds y[i] = v
end

for (name, kernelf) in (("contract", chain_contract!), ("muladd", chain_muladd!))
    bench("performance/pocl/serial/arithmetic/$name") do
        x = uniform(Float64, 8192)
        y = similar(x)
        kernel = kernelf(BACKEND, 256)
        f() = (kernel(y, x, Val(128); ndrange = length(x)); synchronize(BACKEND))
        measure_kernel(f, () -> y ≈ fill(0.5, length(x)))
    end
end

function chain_julia!(y, x, ::Val{n}) where {n}
    @inbounds for i in eachindex(y, x)
        v = x[i]
        for _ in 1:n
            v = muladd(0.5, v, 0.25)
        end
        y[i] = v
    end
    return y
end

bench("performance/julia/serial/arithmetic/muladd-reference") do
    x = Array(uniform(Float64, 8192))
    y = similar(x)
    measure_kernel(() -> chain_julia!(y, x, Val(128)), () -> y ≈ fill(0.5, length(x)))
end


## local memory: a transpose through a tile in local memory, against a direct one

const TILE = 16

@kernel function transpose_tiled!(b, @Const(a))
    gi, gj = @index(Group, NTuple)
    li, lj = @index(Local, NTuple)
    # padded to avoid bank conflicts on GPUs
    tile = @localmem eltype(a) (TILE + 1, TILE)
    @inbounds tile[li, lj] = a[(gi - 1) * TILE + li, (gj - 1) * TILE + lj]
    @synchronize
    @inbounds b[(gj - 1) * TILE + li, (gi - 1) * TILE + lj] = tile[lj, li]
end

@kernel function transpose_naive!(b, @Const(a))
    i, j = @index(Global, NTuple)
    @inbounds b[j, i] = a[i, j]
end

for (name, kernelf) in (("transpose-tiled", transpose_tiled!), ("transpose-naive", transpose_naive!))
    bench("performance/pocl/parallel/localmem/$name") do
        a = uniform(Float32, 1024, 1024)
        b = similar(a)
        kernel = kernelf(BACKEND, (TILE, TILE))
        f() = (kernel(b, a; ndrange = size(a)); synchronize(BACKEND))
        measure_kernel(f, () -> b == permutedims(a))
    end
end


## private state: values of a work-item that are live across `@synchronize` have to be
## kept in memory by PoCL, between the loops over the work-items that it splits a kernel
## into (#893).

const EXCHANGES = 8

@kernel function private_live!(y, @Const(x))
    i = @index(Global, Linear)
    li = @index(Local, Linear)
    n = @uniform prod(@groupsize())
    buf = @localmem Float32 (256,)
    p1 = @inbounds x[i]
    p2, p3, p4 = p1 * p1, p1 + 1.0f0, p1 - 0.5f0
    acc = 0.0f0
    for s in 1:EXCHANGES
        @inbounds buf[li] = acc + p1
        @synchronize
        @inbounds acc = muladd(buf[mod1(li + s, n)], p2 * p3, p4)
        @synchronize
    end
    @inbounds y[i] = acc
end

function private_reference(x)
    y = similar(x)
    acc = zeros(Float32, 256)
    for g in 1:(length(x) ÷ 256)
        xs = x[((g - 1) * 256 + 1):(g * 256)]
        fill!(acc, 0)
        for s in 1:EXCHANGES
            buf = acc .+ xs
            acc = [muladd(buf[mod1(l + s, 256)], xs[l] * xs[l] * (xs[l] + 1.0f0), xs[l] - 0.5f0) for l in 1:256]
        end
        y[((g - 1) * 256 + 1):(g * 256)] = acc
    end
    return y
end

bench("performance/pocl/parallel/private/across-barrier") do
    x = uniform(Float32, 65536)
    y = similar(x)
    kernel = private_live!(BACKEND, 256)
    f() = (kernel(y, x; ndrange = length(x)); synchronize(BACKEND))
    measure_kernel(f, () -> y ≈ private_reference(Array(x)))
end


## collectives: a reduction PoCL supports natively, one of a type it doesn't (which goes
## through shuffles or local memory), and a scan

@kernel function groupsum!(out, @Const(x))
    i = @index(Global, Linear)
    s = @groupreduce(+, x[i], zero(eltype(out)))
    if @index(Local, Linear) == 1
        @inbounds out[@index(Group, Linear)] = s
    end
end

# the sum and maximum of pairs of values
summax(a, b) = (a[1] + b[1], max(a[2], b[2]))

@kernel function groupsummax!(sums, maxs, @Const(x))
    i = @index(Global, Linear)
    v = @inbounds x[i]
    r = @groupreduce(summax, (v, v), (0.0f0, -Inf32))
    if @index(Local, Linear) == 1
        g = @index(Group, Linear)
        @inbounds sums[g], maxs[g] = r
    end
end

@kernel function groupcumsum!(y, @Const(x))
    i = @index(Global, Linear)
    s = @groupscan(+, x[i], zero(eltype(y)))
    @inbounds y[i] = s
end

const COLLECTIVE_SIZE = 2^20
groups(x) = reshape(x, 256, :)

bench("performance/pocl/parallel/groupreduce/add-f64") do
    x = uniform(Float64, COLLECTIVE_SIZE)
    out = KernelAbstractions.zeros(BACKEND, Float64, COLLECTIVE_SIZE ÷ 256)
    kernel = groupsum!(BACKEND, 256)
    f() = (kernel(out, x; ndrange = length(x)); synchronize(BACKEND))
    measure_kernel(f, () -> vec(out) ≈ vec(sum(groups(x); dims = 1)))
end

bench("performance/pocl/parallel/groupreduce/fallback") do
    x = uniform(Float32, COLLECTIVE_SIZE)
    sums = KernelAbstractions.zeros(BACKEND, Float32, COLLECTIVE_SIZE ÷ 256)
    maxs = similar(sums)
    kernel = groupsummax!(BACKEND, 256)
    f() = (kernel(sums, maxs, x; ndrange = length(x)); synchronize(BACKEND))
    measure_kernel(f, () -> vec(sums) ≈ vec(sum(groups(x); dims = 1)) && vec(maxs) == vec(maximum(groups(x); dims = 1)))
end

bench("performance/pocl/parallel/groupscan/add-i32") do
    x = KernelAbstractions.allocate(BACKEND, Int32, COLLECTIVE_SIZE)
    copyto!(x, Int32.(rand(Random.Xoshiro(42), 0:3, COLLECTIVE_SIZE)))
    y = similar(x)
    kernel = groupcumsum!(BACKEND, 256)
    f() = (kernel(y, x; ndrange = length(x)); synchronize(BACKEND))
    measure_kernel(f, () -> groups(y) == cumsum(groups(x); dims = 1))
end


## atomics: a histogram of 256 bins, i.e., contended atomic additions

@kernel function histogram!(bins, @Const(keys))
    i = @index(Global, Linear)
    k = @inbounds keys[i] % UInt32(length(bins)) + 1
    @atomic bins[k] += Int32(1)
end

bench("performance/pocl/parallel/atomics/histogram") do
    keys = KernelAbstractions.allocate(BACKEND, UInt32, 2^20)
    copyto!(keys, rand(Random.Xoshiro(42), UInt32, 2^20))
    bins = KernelAbstractions.zeros(BACKEND, Int32, 256)
    expected = zeros(Int32, 256)
    for k in Array(keys)
        expected[k % 256 + 1] += 1
    end
    kernel = histogram!(BACKEND, 256)
    f() = (kernel(bins, keys; ndrange = length(keys)); synchronize(BACKEND))
    measure_kernel(f, () -> Array(bins) == expected; setup = () -> fill!(bins, 0))
end


## utilities: allocating and initializing an array (on the host up to 1 MiB, with a kernel
## above it, #927), and copying a small array, which launches a kernel

for (name, n) in (("zeros-small", 4096), ("zeros-large", 2^22))
    bench("performance/pocl/parallel/util/$name") do
        f() = KernelAbstractions.zeros(BACKEND, Float32, n)
        x = f()
        length(x) == n && all(iszero, Array(x)) || error("wrong result")
        measure(f, KERNEL)
    end
end

bench("performance/pocl/parallel/util/copyto-small") do
    x = uniform(Float32, 64)
    y = KernelAbstractions.zeros(BACKEND, Float32, 64)
    f() = (KernelAbstractions.copyto!(BACKEND, y, x); synchronize(BACKEND))
    measure_kernel(f, () -> y == x)
end
