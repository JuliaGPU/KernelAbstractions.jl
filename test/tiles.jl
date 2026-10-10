using StaticArrays

# Kernels with tiles whose width is chosen when they are constructed, so that the same
# kernel runs with several widths.

# every tile loops a different number of times, and some return early
@kernel tile function tile_divergent!(out, @Const(x))
    t = @tile()
    i = @index(Global, Linear)
    g = @index(Tile)
    acc = zero(eltype(out))
    for _ in 1:(g % 3)
        acc += tile_reduce(t, +, x[i])
    end
    if g % 4 == 2
        return
    end
    out[i] = acc + tile_shfl(t, x[i], 1)
end

# the tile, its index and its rotation by a lane, the votes, and the ballot of the odd values
# (which needs a sub-group width of at most 64)
@kernel tile function tile_ops!(out, ballots, @Const(x), ::Val{ballot}) where {ballot}
    t = @tile()
    i = @index(Global, Linear)
    out[i, 1] = t.index
    out[i, 2] = t.lane
    out[i, 3] = @index(Tile)
    out[i, 4] = tile_shfl(t, x[i], t.lane + 1)
    out[i, 5] = tile_shfl_up(t, x[i], 1)
    out[i, 6] = tile_shfl_down(t, x[i], 1)
    out[i, 7] = tile_shfl_xor(t, x[i], 1)
    out[i, 8] = tile_any(t, x[i] == 3)
    out[i, 9] = tile_all(t, x[i] > 0)
    out[i, 10] = tile_width(t)
    ballots[i] = ballot ? tile_ballot(t, isodd(x[i])) : zero(UInt64)
end

# code that depends on the width: a mask with a bit per lane, of the smallest type
@inline tile_mask_type(t::Tile) = tile_width(t) <= 8 ? UInt8 : tile_width(t) <= 16 ? UInt16 :
    tile_width(t) <= 32 ? UInt32 : UInt64
@kernel tile function tile_mask!(out, sizes, @Const(x))
    t = @tile()
    i = @index(Global, Linear)
    mask = tile_ballot(t, isodd(x[i])) % tile_mask_type(t)
    out[i] = mask
    sizes[i] = sizeof(mask)
end

# communication through local memory, ordered by `tile_barrier`, with a static workgroup
# size
@kernel tile function tile_barrier!(out, @Const(x))
    t = @tile()
    i = @index(Global, Linear)
    lid = @index(Local, Linear)
    mem = @localmem eltype(out) (prod(@groupsize()),)
    mem[lid] = x[i]
    tile_barrier(t)
    N = tile_width(t)
    out[i] = mem[(t.index - 1) * N + mod1(t.lane + 1, N)]
end

tile_add(a, b) = a .+ b

# reductions of tuples and static arrays
@kernel tile function tile_reduce!(out, out_sv, @Const(x))
    t = @tile()
    i = @index(Global, Linear)
    out[i] = tile_reduce(t, tile_add, (x[i], 2 * x[i]))
    out_sv[i] = tile_reduce(t, +, SVector(x[i], -x[i], Int32(1)))
end

# the composition of affine maps, first `f` then `g`: associative, but not commutative
tile_compose(f, g) = (g[1] * f[1], g[1] * f[2] + g[2])

@kernel tile function tile_scans!(incl, excl, affine, affine_excl, @Const(x), @Const(maps))
    t = @tile()
    i = @index(Global, Linear)
    incl[i] = tile_scan(t, +, x[i])
    excl[i] = tile_exclusive_scan(t, +, x[i], Int32(7))
    affine[i] = tile_scan(t, tile_compose, maps[i])
    affine_excl[i] = tile_exclusive_scan(t, tile_compose, maps[i], (Int32(2), Int32(3)))
end

@kernel tile unsafe_indices = true function tile_unsafe!(out, @Const(x))
    t = @tile()
    i = @index(Global, Linear)
    v = i <= length(x) ? x[i] : zero(eltype(x))
    s = tile_reduce(t, +, v)
    if i <= length(out)
        out[i] = s
    end
end

# a kernel that doesn't compile, to check that invalid launches are rejected before
@kernel tile function tile_uncompilable!(out)
    out[@index(Global, Linear)] = Base.inferencebarrier(identity)(1)
end

# a kernel written for a fixed width
@kernel tile = 8 function tile_fixed!(out)
    t = @tile()
    out[@index(Global, Linear)] = tile_reduce(t, +, Int32(1))
end

# a kernel without tiles
@kernel function tile_plain!(out)
    out[@index(Global, Linear)] = 1
end

@kernel function tile_noargs!()
end

struct TileForward{F}
    f::F
end
(w::TileForward)(args...) = w.f(args...)

# the reference results, by tile of `N` consecutive elements
tile_groups(n, N) = [((i - 1) ÷ N * N + 1):((i - 1) ÷ N * N + N) for i in 1:n]

function tile_testsuite_width(backend, AT, N, tiles)
    W = KI.sub_group_size(backend)
    # several workgroups, and a partial one with a padding tile where possible
    groupsize = tiles == 1 ? N : 2N
    ntiles = 5
    n = ntiles * N
    x = Int32.(rand(1:100, n))
    groups = tile_groups(n, N)

    @test tile_width(tile_divergent!(backend; tile = N)) == N
    @test tile_width(tile_barrier!(backend, groupsize; tile = N)) == N

    divergent = map(1:n) do i
        g = (i - 1) ÷ N + 1
        g % 4 == 2 && return Int32(-1)
        return Int32(g % 3) * sum(x[groups[i]]) + x[first(groups[i])]
    end
    @testset "divergent tiles" begin
        out = AT(fill(Int32(-1), n))
        tile_divergent!(backend; tile = N)(out, AT(x); ndrange = n)
        @test Array(out) == divergent
    end

    @testset "tile operations" begin
        x3 = copy(x)
        x3[2] = 3
        out = AT(zeros(Int32, n, 10))
        ballots = AT(zeros(UInt64, n))
        tile_ops!(backend; tile = N)(out, ballots, AT(x3), Val(W <= 64); ndrange = n)
        out, ballots = Array(out), Array(ballots)
        for i in 1:n
            g = groups[i]
            lane = i - first(g) + 1
            @test out[i, 2] == lane
            @test out[i, 3] == (i - 1) ÷ N + 1
            @test out[i, 4] == x3[g[mod1(lane + 1, N)]]
            @test out[i, 5] == x3[lane > 1 ? i - 1 : i]
            @test out[i, 6] == x3[lane < N ? i + 1 : i]
            @test out[i, 7] == x3[g[((lane - 1) ⊻ 1) + 1]]
            @test out[i, 8] == any(==(3), x3[g])
            @test out[i, 9] == 1
            @test out[i, 10] == N
            if W <= 64
                @test ballots[i] == sum(UInt64(1) << (j - 1) for j in 1:N if isodd(x3[g[j]]); init = UInt64(0))
            end
        end
        # the tile within the workgroup
        @test all(i -> tiles == 1 ? out[i, 1] == 1 : out[i, 1] >= 1, 1:n)
    end

    if W <= 64
        @testset "width-dependent code" begin
            out = AT(zeros(UInt64, n))
            sizes = AT(zeros(Int, n))
            tile_mask!(backend; tile = N)(out, sizes, AT(x); ndrange = n)
            @test all(==(cld(N, 8)), Array(sizes))
            @test Array(out) == [sum(UInt64(1) << (j - 1) for j in 1:N if isodd(x[g[j]]); init = UInt64(0)) for g in groups]
        end
    end

    @testset "tile_barrier" begin
        out = AT(zeros(Int32, n))
        tile_barrier!(backend, groupsize; tile = N)(out, AT(x); ndrange = n)
        @test Array(out) == [x[g[mod1(i - first(g) + 2, N)]] for (i, g) in enumerate(groups)]
    end

    @testset "tuples and static arrays" begin
        out = AT(fill((Int32(0), Int32(0)), n))
        out_sv = AT(fill(SVector{3, Int32}(0, 0, 0), n))
        tile_reduce!(backend; tile = N)(out, out_sv, AT(x); ndrange = n)
        @test Array(out) == [(sum(x[g]), 2 * sum(x[g])) for g in groups]
        @test Array(out_sv) == [SVector(sum(x[g]), -sum(x[g]), Int32(N)) for g in groups]
    end

    @testset "scans" begin
        maps = [(Int32(rand((-1, 1, 2))), Int32(rand(-5:5))) for _ in 1:n]
        incl, excl = AT(zeros(Int32, n)), AT(zeros(Int32, n))
        affine, affine_excl = AT(fill((Int32(0), Int32(0)), n)), AT(fill((Int32(0), Int32(0)), n))
        tile_scans!(backend; tile = N)(incl, excl, affine, affine_excl, AT(x), AT(maps); ndrange = n)
        prefix(f, v, g, i) = foldl(f, v[first(g):i])
        @test Array(incl) == [prefix(+, x, g, i) for (i, g) in enumerate(groups)]
        @test Array(excl) == [i == first(g) ? Int32(7) : Int32(7) + prefix(+, x, g, i - 1) for (i, g) in enumerate(groups)]
        @test Array(affine) == [prefix(tile_compose, maps, g, i) for (i, g) in enumerate(groups)]
        init = (Int32(2), Int32(3))
        @test Array(affine_excl) == [i == first(g) ? init : tile_compose(init, prefix(tile_compose, maps, g, i - 1)) for (i, g) in enumerate(groups)]
    end

    @testset "unsafe_indices" begin
        out = AT(zeros(Int32, n))
        tile_unsafe!(backend; tile = N)(out, AT(x); ndrange = n)
        @test Array(out) == [sum(x[g]) for g in groups]
    end

    @testset "explicit workgroup sizes" begin
        out = AT(fill(Int32(-1), n))
        # a single tile per workgroup also works where several are possible
        tile_divergent!(backend; tile = N)(out, AT(x); ndrange = n, workgroupsize = N)
        @test Array(out) == divergent
        if tiles > 1
            # several tiles per workgroup, which diverge, with a padding tile in the last one
            out = AT(fill(Int32(-1), n))
            tile_divergent!(backend; tile = N)(out, AT(x); ndrange = n, workgroupsize = 2N)
            @test Array(out) == divergent
            out = AT(fill(Int32(-1), n))
            tile_unsafe!(backend; tile = N)(out, AT(x); ndrange = n, workgroupsize = 2N)
            @test Array(out) == [sum(x[g]) for g in groups]
        else
            @test_throws "single tile per workgroup" tile_divergent!(backend; tile = N)(out, AT(x); ndrange = n, workgroupsize = 2N)
            @test_throws "single tile per workgroup" tile_barrier!(backend, 2N; tile = N)
        end
    end

    @testset "rejected launches" begin
        out = AT(zeros(Int32, n))
        k = tile_divergent!(backend; tile = N)
        @test_throws "isn't a multiple of $N" k(out, AT(x); ndrange = n - 1)
        @test_throws "isn't a multiple of $N" k(out, AT(x); ndrange = n, workgroupsize = N ÷ 2)
        @test_throws "isn't a multiple of $N" tile_barrier!(backend, N + N ÷ 2; tile = N)
        @test_throws "isn't 1-D" k(out, AT(x); ndrange = (N, 2))
        @test_throws "isn't 1-D" tile_divergent!(backend, N, (N, 2); tile = N)
        @test_throws "isn't a multiple of $N" tile_divergent!(backend, N, n - 1; tile = N)
        # before compiling the kernel
        u = tile_uncompilable!(backend; tile = N)
        @test_throws "isn't a multiple of $N" u(out; ndrange = n - 1)
        @test_throws "isn't a multiple of $N" u(out; ndrange = n, workgroupsize = N ÷ 2)
        @test_throws "isn't 1-D" u(out; ndrange = (N, 2))
        tiles == 1 && @test_throws "single tile per workgroup" u(out; ndrange = n, workgroupsize = 2N)
        @test_throws "isn't a multiple of $N" u(out; ndrange = n, workgroupsize = 0)
    end
    return
end

function tile_testsuite(backend, AT)
    b = backend()
    @testset "declarations" begin
        @test_throws "power of two" @macroexpand @kernel tile = 3 function f(x)
        end
        @test_throws "need a kernel with tiles" @macroexpand @kernel function f(x)
            t = @tile()
        end
        @test_throws "need a kernel with tiles" @macroexpand @kernel function f(x)
            x[@index(Tile)] = 1
        end
        # all methods of a kernel share its tiles
        @eval @kernel tile function tile_conflict!(x)
        end
        @test_throws "declared with the same tiles" @eval @kernel function tile_conflict!(x, y)
        end
        @test_throws "declared with the same tiles" @eval @kernel tile = 8 function tile_conflict!(x, y)
        end
        # a kernel with tiles in a local scope
        function local_tile_kernel(b)
            @kernel tile = 8 function local_tiles!(out)
                t = @tile()
                out[@index(Global, Linear)] = tile_width(t)
            end
            return local_tiles!(b)
        end
        if tiles_per_workgroup(b, 8) > 0
            @test tile_width(local_tile_kernel(b)) == 8
        end
        @test tile_width(tile_plain!(b)) === nothing
        # also for a kernel without arguments, whose function takes a single argument
        @test KernelAbstractions.tile_policy(tile_noargs!(b).f) === missing
        @test_throws "isn't declared with tiles" tile_plain!(b; tile = 8)
    end
    @test tiles_per_workgroup(b, 3) == 0
    @test tiles_per_workgroup(b, 0) == 0
    if !KI.supports_linear_subgroups(b)
        @test_throws "doesn't support tiles" tile_divergent!(b)
        return
    end
    W = KI.sub_group_size(b)
    @test tiles_per_workgroup(b, 2W) == 0
    @test tiles_per_workgroup(b, W) ==
        (KI.supports_independent_subgroups(b) ? typemax(Int) : 1)

    @testset "construction" begin
        # the default width is the sub-group width
        k = tile_divergent!(b)
        @test tile_width(k) == W
        @test tile_width(tile_divergent!(b, W)) == W
        @test tile_width(tile_divergent!(b, W, 4W)) == W
        @test_throws "power of two" tile_divergent!(b; tile = 3)
        @test_throws "larger than the sub-group width" tile_divergent!(b; tile = 2W)
        # a fixed width
        if W >= 8
            @test tile_width(tile_fixed!(b)) == 8
            @test tile_width(tile_fixed!(b; tile = 8)) == 8
            out = AT(zeros(Int32, 16))
            tile_fixed!(b)(out; ndrange = 16)
            @test all(==(8), Array(out))
        end
        @test_throws "declared with tiles of 8 work-items" tile_fixed!(b; tile = 16)
        # `similar` keeps the width, for a function with compatible tiles
        k16 = tile_divergent!(b; tile = W ÷ 2)
        @test tile_width(similar(k16, tile_ops!(b).f)) == W ÷ 2
        @test_throws "can't run a function declared with `@kernel tile`" similar(tile_plain!(b), k16.f)
        @test tile_width(similar(tile_plain!(b), tile_plain!(b).f)) === nothing
        # a wrapper that forwards its arguments keeps the width
        @test tile_width(similar(k16, TileForward(k16.f))) == W ÷ 2
        @test tile_width(similar(tile_plain!(b), TileForward(tile_plain!(b).f))) === nothing
        # the width reaches the device through the kernel's metadata, also when adapted
        iterspace, _ = KernelAbstractions.partition(k16, (W,), (W ÷ 2,))
        ctx = KernelAbstractions.mkcontext(k16, (W,), iterspace)
        @test KernelAbstractions.__tile_width(ctx) === Val(W ÷ 2)
        @test KernelAbstractions.__tile_width(KernelAbstractions.Adapt.adapt(AT, ctx)) === Val(W ÷ 2)
        @test KernelAbstractions.__tile_width(KernelAbstractions.mkcontext(tile_plain!(b), (W,), iterspace)) === nothing
    end

    # the sub-group width, and a narrower width with one tile per workgroup, with the same
    # kernels
    for N in unique((W, W ÷ 2))
        N >= 2 || continue
        tiles = tiles_per_workgroup(b, N)
        @testset "tile = $N" begin
            tile_testsuite_width(b, AT, N, tiles)
        end
    end
    return
end
