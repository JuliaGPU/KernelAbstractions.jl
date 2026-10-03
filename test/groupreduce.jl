# one result per workgroup, written by every work-item, to check that all of them get it
@kernel function groupreduce_static!(out, @Const(x), op, neutral, ::Val{S}) where {S}
    i = @index(Global, Linear)
    res = @groupreduce(op, x[i], neutral; subgroups = S)
    out[i] = res
end

@kernel function groupreduce_bound!(out, @Const(x), op, neutral, ::Val{N}, ::Val{S}) where {N, S}
    i = @index(Global, Linear)
    res = @groupreduce(op, x[i], neutral, N; subgroups = S)
    out[i] = res
end

# the same call site, and thus local memory, reused in a loop and in a branch
@kernel function groupreduce_loop!(out, @Const(x), ::Val{S}) where {S}
    i = @index(Global, Linear)
    acc = zero(eltype(out))
    for k in 1:3
        res = @groupreduce(+, k * x[i], zero(eltype(out)); subgroups = S)
        acc += res
    end
    if true
        m = @groupreduce max x[i] typemin(eltype(out)) subgroups = S
    end
    out[i] = acc + m
end

@kernel function groupreduce_cartesian!(out, @Const(x), ::Val{S}) where {S}
    I = @index(Global, Cartesian)
    res = @groupreduce(+, x[I], zero(eltype(out)); subgroups = S)
    out[I] = res
end

@kernel unsafe_indices = true function groupreduce_unsafe!(out, @Const(x), ::Val{S}) where {S}
    i = @index(Global, Linear)
    val = i <= length(x) ? x[i] : zero(eltype(out))
    res = @groupreduce(+, val, zero(eltype(out)); subgroups = S)
    if i <= length(out)
        out[i] = res
    end
end

@kernel function subgroupreduce!(out, @Const(x))
    i = @index(Global, Linear)
    res = @subgroupreduce(+, x[i], zero(eltype(out)))
    if KernelInterface.get_sub_group_local_id() == 1
        out[i] = res
    end
end

# the composition of affine maps `x -> a * x + b`, first `f` then `g`: associative, but not
# commutative, so that the scans have to combine the values in order
compose(f, g) = (g[1] * f[1], g[1] * f[2] + g[2])
const affine_identity = (1, 0)

@kernel function groupscan!(out, @Const(x), op, neutral, ::Val{I}) where {I}
    i = @index(Global, Linear)
    res = @groupscan(op, x[i], neutral; inclusive = I)
    out[i] = res
end

@kernel function groupscan_bound!(out, @Const(x), op, neutral, ::Val{N}, ::Val{I}) where {N, I}
    i = @index(Global, Linear)
    res = @groupscan(op, x[i], neutral, N; inclusive = I)
    out[i] = res
end

# the scan in the order of the local linear index, with padding in the middle of a workgroup
@kernel function groupscan_cartesian!(out, @Const(x))
    I = @index(Global, Cartesian)
    res = @groupscan(compose, x[I], affine_identity)
    out[I] = res
end

# the same call site in a loop, and an exclusive scan of the counts
@kernel function groupscan_loop!(out, @Const(x))
    i = @index(Global, Linear)
    acc = 0
    for k in 1:3
        res = @groupscan(+, k * x[i], 0)
        acc += res
    end
    excl = @groupscan (+) x[i] 0 inclusive = false
    out[i] = acc + excl
end

@kernel function subgroupscan!(out, lanes, @Const(x), op, neutral, ::Val{I}) where {I}
    i = @index(Global, Linear)
    res = @subgroupscan(op, x[i], neutral; inclusive = I)
    out[i] = res
    lanes[i] = KernelInterface.get_sub_group_local_id()
end

# reference: the reduction of each workgroup of `groupsize` consecutive elements
function groupwise(op, x, groupsize)
    return [reduce(op, x[((cld(i, groupsize) - 1) * groupsize + 1):min(cld(i, groupsize) * groupsize, end)]) for i in eachindex(x)]
end

# reference: the scan of each workgroup of `groupsize` consecutive elements
function groupwise_scan(op, x, groupsize, neutral, inclusive)
    out = similar(x, typeof(neutral))
    for first in 1:groupsize:length(x)
        group = first:min(first + groupsize - 1, length(x))
        acc = neutral
        for i in group
            inclusive || (out[i] = acc)
            acc = op(acc, x[i])
            inclusive && (out[i] = acc)
        end
    end
    return out
end

function groupreduce_testsuite(backend, AT)
    b = backend()
    algorithms = KI.supports_subgroups(b) ? (false, true) : (false,)

    @testset "subgroups = $S" for S in algorithms
        types = (Int32, Int64, Float32)
        @testset "$T, $(nameof(typeof(op)))" for T in types, (op, neutral) in ((+, zero(T)), (max, typemin(T)))
            S && !KI.supports_shuffle(b, T) && continue
            for (groupsize, n) in ((64, 64), (64, 256), (32, 100), (256, 1000), (7, 23), (1, 3))
                x = T.(rand(1:100, n))
                out = AT(zeros(T, n))
                groupreduce_static!(b, groupsize)(out, AT(x), op, neutral, Val(S); ndrange = n)
                @test Array(out) == groupwise(op, x, groupsize)

                fill!(out, zero(T))
                groupreduce_bound!(b)(out, AT(x), op, neutral, Val(256), Val(S); ndrange = n, workgroupsize = groupsize)
                @test Array(out) == groupwise(op, x, groupsize)
            end
        end

        @testset "loop" begin
            x = rand(1:100, 100)
            out = AT(zeros(Int, 100))
            groupreduce_loop!(b, 64)(out, AT(x), Val(S); ndrange = 100)
            @test Array(out) == 6 .* groupwise(+, x, 64) .+ groupwise(max, x, 64)
        end

        @testset "cartesian" begin
            x = rand(1:100, 10, 12)
            out = AT(zeros(Int, 10, 12))
            groupreduce_cartesian!(b, (4, 8))(out, AT(x), Val(S); ndrange = size(x))
            ref = similar(x)
            for I in CartesianIndices(x)
                g = (cld(I[1], 4) - 1) * 4 .+ (1:4), (cld(I[2], 8) - 1) * 8 .+ (1:8)
                ref[I] = sum(x[intersect(g[1], axes(x, 1)), intersect(g[2], axes(x, 2))])
            end
            @test Array(out) == ref
        end

        @testset "unsafe_indices" begin
            x = rand(1:100, 100)
            out = AT(zeros(Int, 100))
            groupreduce_unsafe!(b, 64)(out, AT(x), Val(S); ndrange = 128)
            @test Array(out) == groupwise(+, x, 64)
        end
    end

    if KI.supports_subgroups(b) && KI.supports_shuffle(b, Float32)
        @testset "@subgroupreduce" begin
            width = KI.sub_group_size(b)
            for (groupsize, n) in ((width, 4width), (2width, 2width + 5))
                x = Float32.(rand(1:100, n))
                out = AT(fill(-1.0f0, n))
                subgroupreduce!(b, groupsize)(out, AT(x); ndrange = n)
                ref = fill(-1.0f0, n)
                for i in 1:width:n
                    ref[i] = sum(x[i:min(i + width - 1, n)])
                end
                @test Array(out) == ref
            end
        end
    end

    @testset "@groupscan" begin
        @testset "inclusive = $I" for I in (true, false)
            for (groupsize, n) in ((64, 64), (64, 256), (32, 100), (256, 1000), (7, 23), (1, 3))
                x = rand(1:100, n)
                out = AT(zeros(Int, n))
                groupscan!(b, groupsize)(out, AT(x), +, 0, Val(I); ndrange = n)
                @test Array(out) == groupwise_scan(+, x, groupsize, 0, I)

                y = [(rand((-1, 1, 2)), rand(-5:5)) for _ in 1:n]
                out = AT(fill((0, 0), n))
                groupscan_bound!(b)(out, AT(y), compose, affine_identity, Val(256), Val(I); ndrange = n, workgroupsize = groupsize)
                @test Array(out) == groupwise_scan(compose, y, groupsize, affine_identity, I)
            end
        end

        @testset "cartesian" begin
            x = [(rand((-1, 1, 2)), rand(-5:5)) for _ in 1:10, _ in 1:12]
            out = AT(fill((0, 0), 10, 12))
            groupscan_cartesian!(b, (4, 8))(out, AT(x); ndrange = size(x))
            ref = similar(x)
            for gi in 1:4:10, gj in 1:8:12
                acc = affine_identity
                # local linear order is column-major within the workgroup
                for j in gj:(gj + 7), i in gi:(gi + 3)
                    (i <= 10 && j <= 12) || continue
                    acc = compose(acc, x[i, j])
                    ref[i, j] = acc
                end
            end
            @test Array(out) == ref
        end

        @testset "loop" begin
            x = rand(1:100, 100)
            out = AT(zeros(Int, 100))
            groupscan_loop!(b, 64)(out, AT(x); ndrange = 100)
            @test Array(out) == 6 .* groupwise_scan(+, x, 64, 0, true) .+ groupwise_scan(+, x, 64, 0, false)
        end
    end

    if KI.supports_subgroups(b) && KI.supports_shuffle(b, Int)
        @testset "@subgroupscan, inclusive = $I" for I in (true, false)
            width = KI.sub_group_size(b)
            # a 1-D workgroup of at most the sub-group width is a single sub-group
            for n in (4width, 2width + 5)
                x = [(rand((-1, 1, 2)), rand(-5:5)) for _ in 1:n]
                out = AT(fill((0, 0), n))
                lanes = AT(zeros(Int, n))
                subgroupscan!(b, width)(out, lanes, AT(x), compose, affine_identity, Val(I); ndrange = n)
                out, lanes = Array(out), Array(lanes)
                # the scan in the order of the lanes; padding work-items contribute the identity
                ref = similar(out)
                for first in 1:width:n
                    group = first:min(first + width - 1, n)
                    for i in group
                        before = filter(j -> I ? lanes[j] <= lanes[i] : lanes[j] < lanes[i], group)
                        sort!(before; by = j -> lanes[j])
                        ref[i] = foldl(compose, x[before]; init = affine_identity)
                    end
                end
                @test out == ref
            end
        end
    end

    @testset "errors" begin
        @test_throws "must be used as a statement" @macroexpand @kernel function f(y, x)
            i = @index(Global)
            y[i] = @groupreduce(+, x[i], 0)
        end
        @test_throws "unknown option" @macroexpand @kernel function f(y, x)
            res = @groupreduce(+, x[1], 0; foo = 1)
        end
    end
    return
end
