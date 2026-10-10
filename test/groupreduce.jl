# one result per workgroup, written by every work-item, to check that all of them get it
@kernel function groupreduce_static!(out, @Const(x), op, neutral)
    i = @index(Global, Linear)
    res = @groupreduce(op, x[i], neutral)
    out[i] = res
end

@kernel function groupreduce_bound!(out, @Const(x), op, neutral, ::Val{N}) where {N}
    i = @index(Global, Linear)
    res = @groupreduce(op, x[i], neutral; groupsize = N)
    out[i] = res
end

# the same call site, and thus local memory, reused in a loop and in a branch
@kernel function groupreduce_loop!(out, @Const(x))
    i = @index(Global, Linear)
    acc = zero(eltype(out))
    for k in 1:3
        res = @groupreduce(+, k * x[i], zero(eltype(out)))
        acc += res
    end
    if true
        m = @groupreduce max x[i] typemin(eltype(out))
    end
    out[i] = acc + m
end

@kernel function groupreduce_cartesian!(out, @Const(x))
    I = @index(Global, Cartesian)
    res = @groupreduce(+, x[I], zero(eltype(out)))
    out[I] = res
end

@kernel unsafe_indices = true function groupreduce_unsafe!(out, @Const(x))
    i = @index(Global, Linear)
    val = i <= length(x) ? x[i] : zero(eltype(out))
    res = @groupreduce(+, val, zero(eltype(out)))
    if i <= length(out)
        out[i] = res
    end
end

# `val` of another type than `neutral`: the padding work-items contribute `neutral`, so the
# value is a `Union` of both types, and the call of the collective must not be union-split
@kernel function groupreduce_mixed!(out, @Const(x))
    i = @index(Global, Linear)
    res = @groupreduce(+, x[i], Int64(0))
    out[i] = res
end

@kernel function groupscan_mixed!(out, @Const(x))
    i = @index(Global, Linear)
    res = @groupscan(+, x[i], 0)
    out[i] = res
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
    res = @groupscan(op, x[i], neutral; groupsize = N, inclusive = I)
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

# A 32-bit float type that no backend shuffles, so that `@groupreduce` uses local memory
# only, also on backends with sub-groups. Its `+` adds the bits as integers.
primitive type TreeFloat <: AbstractFloat 32 end
TreeFloat(x::Integer) = reinterpret(TreeFloat, Int32(x))
Base.:+(a::TreeFloat, b::TreeFloat) = reinterpret(TreeFloat, reinterpret(Int32, a) + reinterpret(Int32, b))
Base.zero(::Type{TreeFloat}) = TreeFloat(0)
Base.typemin(::Type{TreeFloat}) = TreeFloat(typemin(Int32))
Base.:(==)(a::TreeFloat, b::TreeFloat) = reinterpret(Int32, a) == reinterpret(Int32, b)

# Run `f`, and return whether it failed to compile with GPUCompiler.jl#1004 (Metal: a phi of
# a by-reference argument and a device pointer, as in `x[i]` or the `neutral` argument),
# which the local-memory path runs into.
function gpucompiler_1004(f)
    try
        f()
        return false
    catch err
        occursin("Invalid phi record", sprint(showerror, err)) || rethrow()
        return true
    end
end

# `neutral` is evaluated once per work-item, also by the padding work-items
counted_neutral() = 0
count_symbol(ex::Expr, sym) = sum(arg -> count_symbol(arg, sym), ex.args; init = 0)
count_symbol(ex, sym) = Int(ex === sym)

function groupreduce_testsuite(backend, AT)
    b = backend()
    caps = KernelAbstractions.subgroup_capabilities(b)

    @testset "sub-group capabilities" begin
        sub_groups = KI.supports_subgroups(b) && KI.supports_shuffle(b, UInt32)
        @test (caps !== nothing) == sub_groups
        shuffleable(T) = KernelAbstractions.__shuffleable(caps, T)
        @test shuffleable(Int32) == sub_groups
        @test shuffleable(Tuple{Int64, Bool}) == sub_groups
        @test shuffleable(Float32) == (sub_groups && KI.supports_shuffle(b, Float32))
        @test shuffleable(Tuple{Float32, Int32}) == (sub_groups && KI.supports_shuffle(b, Float32))
        @test !shuffleable(TreeFloat)
        @test !shuffleable(Ref{Int})
        # kernels without collectives don't query the capabilities
        @test KernelAbstractions.kernel_subgroups(groupreduce_static!(b, 64)) == caps
        @test KernelAbstractions.kernel_subgroups(groupscan!(b, 64)) == caps
        @test KernelAbstractions.uses_collectives(groupreduce_static!(b, 64).f)
    end

    types = (Int32, Int64, Float32, TreeFloat)
    @testset "$T, $(nameof(typeof(op)))" for T in types, (op, neutral) in ((+, zero(T)), (max, typemin(T)))
        T === TreeFloat && op === max && continue
        for (groupsize, n) in ((64, 64), (64, 256), (32, 100), (256, 1000), (7, 23), (1, 3))
            x = T.(rand(1:100, n))
            out = AT(fill(zero(T), n))
            if gpucompiler_1004(() -> groupreduce_static!(b, groupsize)(out, AT(x), op, neutral; ndrange = n))
                @test_broken false
                continue
            end
            @test Array(out) == groupwise(op, x, groupsize)

            fill!(out, zero(T))
            if gpucompiler_1004(() -> groupreduce_bound!(b)(out, AT(x), op, neutral, Val(256); ndrange = n, workgroupsize = groupsize))
                @test_broken false
                continue
            end
            @test Array(out) == groupwise(op, x, groupsize)
        end
    end

    @testset "argmin" begin
        # (value, index) pairs, the smallest value with the smallest index
        for (groupsize, n) in ((64, 256), (32, 100), (7, 23))
            x = [(Float32(rand(1:20)), Int32(i)) for i in 1:n]
            neutral = (Inf32, typemax(Int32))
            out = AT(fill(neutral, n))
            groupreduce_static!(b, groupsize)(out, AT(x), min, neutral; ndrange = n)
            @test Array(out) == groupwise(min, x, groupsize)
        end
    end

    @testset "loop" begin
        x = rand(1:100, 100)
        out = AT(zeros(Int, 100))
        groupreduce_loop!(b, 64)(out, AT(x); ndrange = 100)
        @test Array(out) == 6 .* groupwise(+, x, 64) .+ groupwise(max, x, 64)
    end

    @testset "cartesian" begin
        x = rand(1:100, 10, 12)
        out = AT(zeros(Int, 10, 12))
        groupreduce_cartesian!(b, (4, 8))(out, AT(x); ndrange = size(x))
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
        groupreduce_unsafe!(b, 64)(out, AT(x); ndrange = 128)
        @test Array(out) == groupwise(+, x, 64)
    end

    @testset "mixed types" begin
        x = Int32.(rand(1:100, 100))
        out = AT(zeros(Int64, 100))
        groupreduce_mixed!(b, 64)(out, AT(x); ndrange = 100)
        @test Array(out) == groupwise(+, Int64.(x), 64)
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

        @testset "mixed types" begin
            x = Int32.(rand(1:100, 100))
            out = AT(zeros(Int, 100))
            groupscan_mixed!(b, 64)(out, AT(x); ndrange = 100)
            @test Array(out) == groupwise_scan(+, Int.(x), 64, 0, true)
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
        @test_throws "given more than once" @macroexpand @kernel function f(y, x)
            res = @groupreduce(+, x[1], 0; groupsize = 32, groupsize = 64)
        end
        # the upper bound of the workgroup size is a keyword
        @test_throws "expects `op`, `val` and `neutral`" @macroexpand @kernel function f(y, x)
            res = @groupreduce(+, x[1], 0, 64)
        end
        @test_throws "unknown option" @macroexpand @kernel function f(y, x)
            res = @groupreduce(+, x[1], 0; subgroups = true)
        end
    end

    @testset "neutral evaluated once" begin
        ex = @macroexpand @kernel function f(y, x)
            i = @index(Global, Linear)
            res = @groupreduce(+, x[i], counted_neutral())
            y[i] = res
        end
        @test count_symbol(ex, :counted_neutral) == 1
    end
    return
end
