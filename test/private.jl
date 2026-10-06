using KernelAbstractions
using StaticArrays
using Test

@kernel function stmt_form()
    @uniform bs = @groupsize()[1]
    @private s = bs ÷ 2
    @synchronize
end

@kernel function typetest(A, B)
    priv = @private eltype(A) (1,)
    I = @index(Global, Linear)
    @inbounds begin
        B[I] = eltype(priv) === eltype(A)
    end
end

@kernel function private(A)
    @uniform N = prod(@groupsize())
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    priv = @private Int (1,)
    @inbounds begin
        priv[1] = N - i + 1
        @synchronize
        A[I] = priv[1]
    end
end

# This is horrible don't write code like this
@kernel function forloop(A, ::Val{N}) where {N}
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    priv = @private Int (N,)
    for j in 1:N
        priv[j] = A[I, j]
    end
    A[I, 1] = 0
    @synchronize
    for j in 1:N
        k = mod1(j + i - 1, N)
        A[k, 1] += priv[j]
        @synchronize
    end
end

@kernel function reduce_private(out, A)
    I = @index(Global, NTuple)
    i = @index(Local)

    priv = @private eltype(A) (1,)
    @inbounds begin
        priv[1] = zero(eltype(A))
        for k in 1:size(A, ndims(A))
            priv[1] += A[I..., k]
        end
        out[I...] = priv[1]
    end
end

# whole-array reductions, which use StaticArrays' implementations, compared against the same
# code on an `Array`
@kernel function private_reductions(out, A)
    I = @index(Global, Linear)
    priv = @private Float32 (8,)
    for j in 1:8
        @inbounds priv[j] = A[I] * j - 4
    end
    @inbounds begin
        out[I, 1] = sum(priv)
        out[I, 2] = sum(abs2, priv)
        out[I, 3] = prod(priv)
        out[I, 4] = maximum(priv)
        out[I, 5] = minimum(abs, priv)
        out[I, 6] = last(extrema(priv))
        out[I, 7] = reduce(+, priv; init = 100.0f0)
        out[I, 8] = foldl((a, b) -> 2a - b, priv)
        out[I, 9] = foldr((a, b) -> 2a - b, priv)
        out[I, 10] = mapreduce(abs, max, priv)
        out[I, 11] = any(x -> x > 3 * A[I], priv)
        out[I, 12] = all(x -> x > -4, priv)
        out[I, 13] = count(x -> x > 0, priv)
    end
end

function private_reductions_ref(a)
    priv = Float32[a * j - 4 for j in 1:8]
    return Float32[
        sum(priv), sum(abs2, priv), prod(priv), maximum(priv), minimum(abs, priv),
        last(extrema(priv)), reduce(+, priv; init = 100.0f0),
        foldl((a, b) -> 2a - b, priv), foldr((a, b) -> 2a - b, priv),
        mapreduce(abs, max, priv), any(x -> x > 3 * a, priv), all(x -> x > -4, priv),
        count(x -> x > 0, priv),
    ]
end

# static-array operations that return a new array give a static array, not a heap `Array`
@kernel function private_static(out, A)
    I = @index(Global, Linear)
    p = @private Float32 (8,)
    q = @private Float32 (8,)
    for j in 1:8
        @inbounds p[j] = A[I] * j
    end
    q .= p .+ 1
    @inbounds begin
        r = p + q
        out[I, 1] = r[3]
        out[I, 2] = (p + q)[:][8]
        out[I, 3] = sum(p[StaticArrays.SUnitRange(2, 7)])
        out[I, 4] = (p + SVector{8}(ntuple(_ -> 1.0f0, 8)))[5]
        out[I, 5] = map(abs2, p)[2]
        out[I, 6] = (p .* 2.0f0)[4]
        s = SVector(Tuple(p))
        p[1] = 0
        out[I, 7] = s[1]
    end
end

function private_static_ref(a)
    p = Float32[a * j for j in 1:8]
    q = p .+ 1
    return Float32[(p + q)[3], (p + q)[8], sum(p[2:7]), p[5] + 1, p[2]^2, 2 * p[4], p[1]]
end

@noinline private_consume(p) = @inbounds p[1] + p[end]

# in-place operations, views, aliasing on assignment, and passing to a non-inlined function
@kernel function private_inplace(out, A)
    I = @index(Global, Linear)
    p = @private Float32 (2, 4)
    q = p
    @inbounds begin
        fill!(p, A[I])
        q[2, 3] = 0
        v = view(p, :, 4)
        v .*= 2
        out[I, 1] = p[2, 3]
        out[I, 2] = p[1, 4]
        out[I, 3] = private_consume(p)
    end
end

function private_testsuite(backend, ArrayT)
    @testset "kernels" begin
        stmt_form(backend(), 16)(ndrange = 16)
        synchronize(backend())
        A = ArrayT{Int}(undef, 64)
        private(backend(), 16)(A, ndrange = size(A))
        synchronize(backend())
        B = Array(A)
        @test all(B[1:16] .== 16:-1:1)
        @test all(B[17:32] .== 16:-1:1)
        @test all(B[33:48] .== 16:-1:1)
        @test all(B[49:64] .== 16:-1:1)

        A = ArrayT{Int}(undef, 64, 64)
        A .= 1
        forloop(backend())(A, Val(size(A, 2)), ndrange = size(A, 1), workgroupsize = size(A, 1))
        synchronize(backend())
        @test all(Array(A)[:, 1] .== 64)
        @test all(Array(A)[:, 2:end] .== 1)

        B = ArrayT{Bool}(undef, size(A)...)
        typetest(backend(), 16)(A, B, ndrange = size(A))
        synchronize(backend())
        @test all(Array(B))

        A = ArrayT(ones(Float32, 64, 3))
        out = ArrayT{Float32}(undef, 64)
        reduce_private(backend(), 8)(out, A, ndrange = size(out))
        synchronize(backend())
        @test all(Array(out) .== 3.0f0)
    end

    @testset "reductions" begin
        A = ArrayT(Float32.(1:64) ./ 8)
        out = ArrayT{Float32}(undef, 64, 13)
        private_reductions(backend(), 16)(out, A, ndrange = 64)
        synchronize(backend())
        ref = reduce(hcat, private_reductions_ref.(Array(A)))'
        @test Array(out) ≈ ref
    end

    @testset "static arrays" begin
        A = ArrayT(Float32.(1:64) ./ 8)
        out = ArrayT{Float32}(undef, 64, 7)
        private_static(backend(), 16)(out, A, ndrange = 64)
        synchronize(backend())
        @test Array(out) ≈ reduce(hcat, private_static_ref.(Array(A)))'
    end

    @testset "host" begin
        # the storage is an alloca that only device code can lower
        @test_throws MethodError KernelAbstractions.Scratchpad(nothing, Float32, Val((8,)))
    end

    @testset "in-place" begin
        A = ArrayT(Float32.(1:64))
        out = ArrayT{Float32}(undef, 64, 3)
        private_inplace(backend(), 16)(out, A, ndrange = 64)
        synchronize(backend())
        a = Array(A)
        @test Array(out) == [zero(a) 2a 3a]
    end

    return
end
