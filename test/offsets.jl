using KernelAbstractions
using KernelAbstractions.NDIteration
using Test

@kernel function offsets_fill_indices!(out, lo)
    I = @index(Global, NTuple)
    i = @index(Global, Linear)
    @inbounds out[(I .- lo .+ 1)...] = i
end

@kernel function offsets_fill_ndrange!(out, lo)
    I = @index(Global, NTuple)
    sz = @ndrange()
    @inbounds out[(I .- lo .+ 1)...] = sz[1]
end

@kernel function offsets_fill_cartesian!(out, lo)
    I = @index(Global, Cartesian)
    @inbounds out[I - lo + oneunit(I)] = 1
end

function offsets_testsuite(Backend, AT)
    backend = Backend()
    ranges = (-3:4, 2:11)
    lo = map(first, ranges)
    ref = reshape(1:80, 8, 10)
    fresh() = AT(zeros(Int, 8, 10))

    @testset "static ndrange" begin
        out = fresh()
        offsets_fill_indices!(backend, (4, 4), ranges)(out, lo)
        synchronize(backend)
        @test Array(out) == ref
    end

    @testset "dynamic ndrange" begin
        out = fresh()
        offsets_fill_indices!(backend, (4, 4))(out, lo; ndrange = ranges)
        synchronize(backend)
        @test Array(out) == ref

        out = fresh()
        offsets_fill_indices!(backend, (4, 4))(out, lo; ndrange = CartesianIndices(ranges))
        synchronize(backend)
        @test Array(out) == ref

        out = fresh()
        offsets_fill_indices!(backend)(out, lo; ndrange = ranges, workgroupsize = (4, 4))
        synchronize(backend)
        @test Array(out) == ref
    end

    @testset "mixed extents and ranges" begin
        out = fresh()
        offsets_fill_indices!(backend, (4, 4))(out, (1, 2); ndrange = (8, 2:11))
        synchronize(backend)
        @test Array(out) == ref
    end

    @testset "ragged workgroups" begin
        out = fresh()
        offsets_fill_indices!(backend, (3, 3))(out, lo; ndrange = ranges)
        synchronize(backend)
        @test Array(out) == ref
    end

    @testset "bare range" begin
        out = AT(zeros(Int, 16))
        offsets_fill_indices!(backend, 4)(out, (5,); ndrange = 5:20)
        synchronize(backend)
        @test Array(out) == 1:16
    end

    @testset "cartesian index" begin
        out = fresh()
        offsets_fill_cartesian!(backend, (4, 4))(out, CartesianIndex(lo); ndrange = ranges)
        synchronize(backend)
        @test all(==(1), Array(out))
    end

    @testset "@ndrange returns extents" begin
        out = fresh()
        offsets_fill_ndrange!(backend, (4, 4))(out, lo; ndrange = ranges)
        synchronize(backend)
        @test all(==(8), Array(out))
    end

    @testset "empty range" begin
        out = fresh()
        offsets_fill_indices!(backend, (4, 4))(out, lo; ndrange = (5:4, 1:3))
        synchronize(backend)
        @test all(iszero, Array(out))
    end
    return
end
