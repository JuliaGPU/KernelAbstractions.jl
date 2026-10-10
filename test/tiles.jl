using StaticArrays

# Kernels with tiles of `N` work-items, for the tile widths that backends can have (the
# width is a constant of the kernel, so every width needs its own kernels).
const TILE_WIDTHS = (8, 16, 32, 64)
const tile_kernels = Dict{Int, Any}()

tile_add(a, b) = a .+ b

for N in TILE_WIDTHS
    # every tile loops a different number of times, and some return early
    divergent = Symbol(:tile_divergent_, N)
    @eval @kernel tile = $N function $divergent(out, @Const(x))
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

    # the tile, its index and its rotation by a lane, and the ballot of the odd values
    ops = Symbol(:tile_ops_, N)
    # (the ballot needs a sub-group width of at most 64)
    @eval @kernel tile = $N function $ops(out, ballots, @Const(x), ::Val{ballot}) where {ballot}
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
        ballots[i] = ballot ? tile_ballot(t, isodd(x[i])) : zero(UInt64)
    end

    # communication through local memory, ordered by `tile_barrier`, with a static workgroup
    # size; the tile operations in a helper function
    barrier = Symbol(:tile_barrier_, N)
    @eval @kernel tile = $N function $barrier(out, @Const(x))
        t = @tile()
        i = @index(Global, Linear)
        lid = @index(Local, Linear)
        mem = @localmem eltype(out) (prod(@groupsize()),)
        mem[lid] = x[i]
        tile_barrier(t)
        out[i] = mem[(t.index - 1) * $N + mod1(t.lane + 1, $N)]
    end

    # reductions of tuples and static arrays
    reduce = Symbol(:tile_reduce_, N)
    @eval @kernel tile = $N function $reduce(out, out_sv, @Const(x))
        t = @tile()
        i = @index(Global, Linear)
        out[i] = tile_reduce(t, tile_add, (x[i], 2 * x[i]))
        out_sv[i] = tile_reduce(t, +, SVector(x[i], -x[i], Int32(1)))
    end

    unsafe = Symbol(:tile_unsafe_, N)
    @eval @kernel tile = $N unsafe_indices = true function $unsafe(out, @Const(x))
        t = @tile()
        i = @index(Global, Linear)
        v = i <= length(x) ? x[i] : zero(eltype(x))
        s = tile_reduce(t, +, v)
        if i <= length(out)
            out[i] = s
        end
    end

    # a kernel that doesn't compile, to check that invalid launches are rejected before
    uncompilable = Symbol(:tile_uncompilable_, N)
    @eval @kernel tile = $N function $uncompilable(out)
        out[@index(Global, Linear)] = Base.inferencebarrier(identity)(1)
    end

    tile_kernels[N] = (;
        divergent = eval(divergent), ops = eval(ops), barrier = eval(barrier),
        reduce = eval(reduce), unsafe = eval(unsafe), uncompilable = eval(uncompilable),
    )
end

# the reference results, by tile of `N` consecutive elements
tile_groups(n, N) = [((i - 1) ÷ N * N + 1):((i - 1) ÷ N * N + N) for i in 1:n]

function tile_testsuite_width(backend, AT, N, tiles)
    k = tile_kernels[N]
    W = KI.sub_group_size(backend)
    # several workgroups, and a partial one with a padding tile where possible
    groupsize = tiles == 1 ? N : 2N
    ntiles = 5
    n = ntiles * N
    x = Int32.(rand(1:100, n))
    groups = tile_groups(n, N)

    divergent = map(1:n) do i
        g = (i - 1) ÷ N + 1
        g % 4 == 2 && return Int32(-1)
        return Int32(g % 3) * sum(x[groups[i]]) + x[first(groups[i])]
    end
    @testset "divergent tiles" begin
        out = AT(fill(Int32(-1), n))
        k.divergent(backend)(out, AT(x); ndrange = n)
        @test Array(out) == divergent
    end

    @testset "tile operations" begin
        x3 = copy(x)
        x3[2] = 3
        out = AT(zeros(Int32, n, 9))
        ballots = AT(zeros(UInt64, n))
        k.ops(backend)(out, ballots, AT(x3), Val(W <= 64); ndrange = n)
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
            if W <= 64
                @test ballots[i] == sum(UInt64(1) << (j - 1) for j in 1:N if isodd(x3[g[j]]); init = UInt64(0))
            end
        end
        # the tile within the workgroup
        @test all(i -> tiles == 1 ? out[i, 1] == 1 : out[i, 1] >= 1, 1:n)
    end

    @testset "tile_barrier" begin
        out = AT(zeros(Int32, n))
        k.barrier(backend, groupsize)(out, AT(x); ndrange = n)
        @test Array(out) == [x[g[mod1(i - first(g) + 2, N)]] for (i, g) in enumerate(groups)]
    end

    @testset "tuples and static arrays" begin
        out = AT(fill((Int32(0), Int32(0)), n))
        out_sv = AT(fill(SVector{3, Int32}(0, 0, 0), n))
        k.reduce(backend)(out, out_sv, AT(x); ndrange = n)
        @test Array(out) == [(sum(x[g]), 2 * sum(x[g])) for g in groups]
        @test Array(out_sv) == [SVector(sum(x[g]), -sum(x[g]), Int32(N)) for g in groups]
    end

    @testset "unsafe_indices" begin
        out = AT(zeros(Int32, n))
        k.unsafe(backend)(out, AT(x); ndrange = n)
        @test Array(out) == [sum(x[g]) for g in groups]
    end

    @testset "explicit workgroup sizes" begin
        out = AT(fill(Int32(-1), n))
        # a single tile per workgroup also works where several are possible
        k.divergent(backend)(out, AT(x); ndrange = n, workgroupsize = N)
        @test Array(out) == divergent
        if tiles > 1
            # several tiles per workgroup, which diverge, with a padding tile in the last one
            out = AT(fill(Int32(-1), n))
            k.divergent(backend)(out, AT(x); ndrange = n, workgroupsize = 2N)
            @test Array(out) == divergent
            out = AT(fill(Int32(-1), n))
            k.unsafe(backend)(out, AT(x); ndrange = n, workgroupsize = 2N)
            @test Array(out) == [sum(x[g]) for g in groups]
        else
            @test_throws "single tile per workgroup" k.divergent(backend)(out, AT(x); ndrange = n, workgroupsize = 2N)
            @test_throws "single tile per workgroup" k.barrier(backend, 2N)
        end
    end

    @testset "rejected launches" begin
        out = AT(zeros(Int32, n))
        @test_throws "isn't a multiple of $N" k.divergent(backend)(out, AT(x); ndrange = n - 1)
        N > 1 && @test_throws "isn't a multiple of $N" k.divergent(backend)(out, AT(x); ndrange = n, workgroupsize = N ÷ 2)
        N > 1 && @test_throws "isn't a multiple of $N" k.barrier(backend, N + N ÷ 2)
        @test_throws "isn't 1-D" k.divergent(backend)(out, AT(x); ndrange = (N, 2))
        # before compiling the kernel
        @test_throws "isn't a multiple of $N" k.uncompilable(backend)(out; ndrange = n - 1)
        N > 1 && @test_throws "isn't a multiple of $N" k.uncompilable(backend)(out; ndrange = n, workgroupsize = N ÷ 2)
        @test_throws "isn't 1-D" k.uncompilable(backend)(out; ndrange = (N, 2))
        tiles == 1 && @test_throws "single tile per workgroup" k.uncompilable(backend)(out; ndrange = n, workgroupsize = 2N)
        @test_throws "isn't a multiple of $N" k.uncompilable(backend)(out; ndrange = n, workgroupsize = 0)
    end
    return
end

function tile_testsuite(backend, AT)
    b = backend()
    @testset "errors" begin
        @test_throws "power of two" @macroexpand @kernel tile = 3 function f(x)
        end
        @test_throws "need a kernel with tiles" @macroexpand @kernel function f(x)
            t = @tile()
        end
        @test_throws "need a kernel with tiles" @macroexpand @kernel function f(x)
            x[@index(Tile)] = 1
        end
    end
    @test tiles_per_workgroup(b, 3) == 0
    @test tiles_per_workgroup(b, 0) == 0
    KI.supports_linear_subgroups(b) || return
    W = KI.sub_group_size(b)
    @test tiles_per_workgroup(b, 2W) == 0
    @test tiles_per_workgroup(b, W) ==
        (KI.supports_independent_subgroups(b) ? typemax(Int) : 1)

    # the sub-group width, and a narrower width with one tile per workgroup
    for N in (W, W ÷ 2)
        N in TILE_WIDTHS || continue
        tiles = tiles_per_workgroup(b, N)
        @testset "tile = $N" begin
            tile_testsuite_width(b, AT, N, tiles)
        end
    end
    return
end
