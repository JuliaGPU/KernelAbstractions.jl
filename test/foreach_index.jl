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
    foreach_index(dst, src) do i
        @inbounds dst[i] = src[i]
    end
    return dst
end

function foreach_index_copy_from_first!(src, dst)
    foreach_index(src, dst) do i
        @inbounds dst[i] = src[i]
    end
    return dst
end

function foreach_index_copy_backend!(dst, src, backend)
    foreach_index(backend, eachindex(src)) do i
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
    foreach_index(backend, range) do i
        @inbounds @atomic out[1] += i % eltype(out)
    end
    return out
end

# Record every index of the index space `indices` in `out`, an array indexed from 1.
function foreach_index_record!(out, backend, indices; workgroupsize = nothing)
    offset = first(indices) - oneunit(first(indices))
    foreach_index(backend, indices; workgroupsize) do I
        @inbounds out[I - offset] = I
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
        # (32-bit atomics, as not every backend has 64-bit ones)
        out = AT(zeros(Int32, 1))
        foreach_index_sum_range!(out, 1:100, backend)
        synchronize(backend)
        @test Array(out)[1] == sum(1:100)
    end

    @testset "index space as given" begin
        # the values of the index space, not its own indices
        @testset "$(indices)" for indices in (
                5:20, -3:4, Base.IdentityUnitRange(-3:4), Int32(2):Int32(9), UInt(2):UInt(9),
            )
            out = AT(zeros(Int, length(indices)))
            foreach_index_record!(out, backend, indices)
            synchronize(backend)
            @test Array(out) == collect(indices)
        end

        # e.g. the interior of an array, with default and explicit workgroup sizes
        @testset "$(indices), workgroupsize=$(workgroupsize)" for indices in (
                    CartesianIndices((2:9, 2:7)), CartesianIndices((-1:3, 0:0, 4:6)),
                ), workgroupsize in (nothing, 4, (4, 4))
            out = AT(fill(CartesianIndex(ntuple(_ -> 0, ndims(indices))), size(indices)))
            foreach_index_record!(out, backend, indices; workgroupsize)
            synchronize(backend)
            @test Array(out) == collect(indices)
        end

        # indices beyond the range of `Int32`
        indices = (typemax(Int32) + 1):(typemax(Int32) + 10)
        out = AT(zeros(Int, 10))
        foreach_index_record!(out, backend, indices)
        synchronize(backend)
        @test Array(out) == collect(indices)
    end

    @testset "several arrays" begin
        # linear indices if all arrays have them
        A = AT(collect(reshape(1:12, 3, 4)))
        B = AT(zeros(Int, 3, 4))
        foreach_index_copy!(B, A)
        synchronize(backend)
        @test Array(B) == Array(A)

        # Cartesian indices otherwise
        C = AT(zeros(Int, 6, 4))
        v = view(C, 1:2:6, :)
        @test eachindex(v, A) isa CartesianIndices
        foreach_index_copy!(v, A)
        synchronize(backend)
        @test Array(C)[1:2:6, :] == Array(A)
        @test all(iszero, Array(C)[2:2:6, :])

        # arrays whose indices differ
        @test_throws DimensionMismatch foreach_index_copy!(AT(zeros(Int, 4, 3)), view(C, 1:2:6, :))

        # arrays on different backends
        if get_backend(AT(Int[])) != get_backend(Int[])
            @test_throws ArgumentError foreach_index_copy!(AT(zeros(Int, 3)), zeros(Int, 3))
        end

        # ranges, and views of them, run on the backend of the other arrays
        for r in (Base.OneTo(3), 4:6, view(reshape(1:12, 3, 4), :, 2))
            dst = AT(zeros(Int, 3))
            foreach_index_copy!(dst, r)
            synchronize(backend)
            @test Array(dst) == collect(r)
        end
        dst = AT(zeros(Int, 2, 3))
        foreach_index_copy!(dst, reshape(1:6, 2, 3))
        synchronize(backend)
        @test Array(dst) == reshape(1:6, 2, 3)
        dst = AT(zeros(Int, 2, 3))
        foreach_index_copy!(dst, LinearIndices((2, 3)))
        synchronize(backend)
        @test Array(dst) == collect(LinearIndices((2, 3)))
        # ... also as the first argument
        dst = AT(zeros(Int, 3))
        foreach_index_copy_from_first!(2:4, dst)
        synchronize(backend)
        @test Array(dst) == 2:4
        # ... and need one if there are no other arrays
        @test_throws "pass it explicitly" foreach_index_copy!(1:3, 4:6)
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

    @testset "zero-dimensional index space" begin
        # a single index
        @testset "workgroupsize=$(workgroupsize)" for workgroupsize in (nothing, 4)
            out = AT(fill(CartesianIndex(1), 1))
            foreach_index(backend, CartesianIndices(()); workgroupsize) do I
                @inbounds out[1] = CartesianIndex(length(I) + 2)
            end
            synchronize(backend)
            @test Array(out) == [CartesianIndex(2)]
        end

        A = AT(fill(1))
        foreach_index_fill_index!(A)
        synchronize(backend)
        @test Array(A)[] == 1
    end

    @testset "empty index space" begin
        src = AT(Int[])
        dst = AT(Int[])
        @test foreach_index_copy!(dst, src) === dst
        synchronize(backend)
        @test isempty(Array(dst))

        out = AT(zeros(Int32, 1))
        foreach_index_sum_range!(out, 5:4, backend)
        foreach_index_record!(out, backend, CartesianIndices((1:2, 3:2)))
        synchronize(backend)
        @test Array(out) == [0]
    end

    @testset "errors" begin
        # index spaces that an `ndrange` cannot express, also when they are empty
        @test_throws ArgumentError foreach_index(identity, backend, [1, 2, 3])
        @test_throws ArgumentError foreach_index(identity, backend, Int[])
        @test_throws ArgumentError foreach_index(identity, backend, 1:2:9)
        @test_throws ArgumentError foreach_index(identity, backend, CartesianIndices((1:2:9, 1:3)))
        @test_throws ArgumentError foreach_index(identity, backend, CartesianIndices((1:2:1, 1:3)))
        @test_throws ArgumentError foreach_index(identity, backend, Dict(1 => 2))

        # a function capturing a boxed variable
        s = 1
        boxed = i -> s
        s = 2
        @test_throws "reassigned (`s`)" foreach_index(boxed, backend, 1:3)

        # a collection that is not an array has no backend, and needs the second form
        @test_throws MethodError foreach_index(identity, (1, 2, 3))
    end
    return
end
