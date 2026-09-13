using KernelAbstractions
using KernelAbstractions: @atomic
using Adapt
using Test

# An `IndexLinear` array whose `axes1` does not start at 1, so that `eachindex` returns a
# range of linear indices with an offset.
struct OffsetVec{T, A <: AbstractVector{T}} <: AbstractVector{T}
    data::A
    r::Base.IdentityUnitRange{UnitRange{Int}}
end
OffsetVec(data::AbstractVector, off::Int) = OffsetVec(data, Base.IdentityUnitRange((1 + off):(length(data) + off)))
Base.size(v::OffsetVec) = size(v.data)
Base.axes(v::OffsetVec) = (v.r,)
Base.IndexStyle(::Type{<:OffsetVec}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(v::OffsetVec, i::Int) = v.data[i - first(v.r) + 1]
Base.@propagate_inbounds Base.setindex!(v::OffsetVec, x, i::Int) = (v.data[i - first(v.r) + 1] = x)
KernelAbstractions.get_backend(v::OffsetVec) = KernelAbstractions.get_backend(v.data)
Adapt.adapt_structure(to, v::OffsetVec) = OffsetVec(adapt(to, v.data), v.r)

# The loop bodies live in functions so that their captures have known types, as the
# `foreach_index` docstring requires.

function foreach_index_copy!(dst, src)
    foreach_index(src) do i
        @inbounds dst[i] = src[i]
    end
    return dst
end

function foreach_index_copy_backend!(dst, src, backend)
    foreach_index(src, backend) do i
        @inbounds dst[i] = src[i]
    end
    return dst
end

function foreach_index_copy_workgroupsize!(dst, src, workgroupsize)
    foreach_index(src; workgroupsize) do i
        @inbounds dst[i] = src[i]
    end
    return dst
end

function foreach_index_fill_index!(v)
    foreach_index(v) do i
        @inbounds v[i] = i
    end
    return v
end

function foreach_index_mark!(out, itr)
    foreach_index(itr) do I
        @inbounds out[I] += 1
    end
    return out
end

function foreach_index_sum_range!(out, range, backend)
    foreach_index(range, backend) do i
        @inbounds @atomic out[1] += i
    end
    return out
end

function foreach_index_testsuite(Backend, AT)
    backend = Backend()

    @testset "linear index space" begin
        @testset "$(length(dims))-D" for dims in ((1024,), (32, 33), (4, 5, 6))
            src = AT(reshape(collect(1:prod(dims)), dims))
            dst = AT(zeros(Int, dims))
            @test foreach_index_copy!(dst, src) === dst
            synchronize(backend)
            @test Array(dst) == Array(src)

            # every index is visited exactly once
            out = AT(zeros(Int, dims))
            foreach_index_mark!(out, src)
            synchronize(backend)
            @test all(isone, Array(out))
        end
    end

    @testset "cartesian index space" begin
        A = AT(zeros(Int, 8, 10))
        v = view(A, 1:2:8, 2:2:10)
        @test eachindex(v) isa CartesianIndices
        foreach_index_mark!(v, v)
        synchronize(backend)
        ref = zeros(Int, 8, 10)
        ref[1:2:8, 2:2:10] .= 1
        @test Array(A) == ref
    end

    @testset "offset linear index space" begin
        v = OffsetVec(AT(zeros(Int, 10)), -3)
        @test eachindex(v) == -2:7
        foreach_index_fill_index!(v)
        synchronize(backend)
        @test Array(v.data) == collect(-2:7)
    end

    @testset "index space without device memory" begin
        # a range carries no backend of its own, so it has to be given
        out = AT(zeros(Int, 1))
        foreach_index_sum_range!(out, 1:100, backend)
        synchronize(backend)
        @test Array(out)[1] == sum(1:100)
    end

    @testset "workgroupsize" begin
        src = AT(collect(1:1000))
        dst = AT(zeros(Int, 1000))
        foreach_index_copy_workgroupsize!(dst, src, 8)
        synchronize(backend)
        @test Array(dst) == Array(src)
    end

    @testset "explicit backend" begin
        src = AT(collect(1:100))
        dst = AT(zeros(Int, 100))
        foreach_index_copy_backend!(dst, src, backend)
        synchronize(backend)
        @test Array(dst) == Array(src)
    end

    @testset "empty index space" begin
        src = AT(Int[])
        dst = AT(Int[])
        @test foreach_index_copy!(dst, src) === dst
        synchronize(backend)
        @test isempty(Array(dst))
    end

    @testset "errors" begin
        # an index space that is neither linear nor cartesian
        @test_throws ArgumentError foreach_index(identity, Dict(1 => 2), backend)
    end
    return
end
