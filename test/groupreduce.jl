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

# reference: the reduction of each workgroup of `groupsize` consecutive elements
function groupwise(op, x, groupsize)
    return [reduce(op, x[((cld(i, groupsize) - 1) * groupsize + 1):min(cld(i, groupsize) * groupsize, end)]) for i in eachindex(x)]
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
