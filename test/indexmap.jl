using KernelAbstractions
using KernelAbstractions.NDIteration
using Test

@kernel function indexmap_mark!(A, count)
    I = @index(Global, Cartesian)
    p = @index(Global, Linear)
    @inbounds A[I] += 1
    @inbounds count[p] = p
end

@kernel function indexmap_mark_ntuple!(A)
    i, j, k = @index(Global, NTuple)
    @inbounds A[i, j, k] += 1
end

@kernel function indexmap_positions!(count)
    p = @index(Global, Linear)
    g = @index(Group, Linear)
    l = @index(Local, Linear)
    nd = @ndrange()
    @inbounds count[p] = p + 1000 * g + 100000 * l + 10^7 * nd[1]
end

function indexmap_testsuite(Backend, AT)
    backend = Backend()
    dims = (4, 5, 6)
    indices = [CartesianIndex(i, j, k) for i in 1:4, j in 1:5, k in 1:6 if (i + j + k) % 3 == 0]
    n = length(indices)
    ref = zeros(Int, dims)
    for I in indices
        ref[I] = 1
    end

    @testset "$(eltype(map))" for map in (indices, Tuple.(indices), [Int32.(Tuple(I)) for I in indices])
        A = AT(zeros(Int, dims))
        count = AT(zeros(Int, n))
        indexmap_mark!(backend, 4)(A, count; ndrange = AT(map))
        synchronize(backend)
        @test Array(A) == ref
        @test Array(count) == 1:n

        A = AT(zeros(Int, dims))
        indexmap_mark_ntuple!(backend)(A; ndrange = AT(map), workgroupsize = 8)
        synchronize(backend)
        @test Array(A) == ref
    end

    @testset "group and local indices" begin
        count = AT(zeros(Int, n))
        indexmap_positions!(backend, 4)(count; ndrange = AT(indices))
        synchronize(backend)
        @test Array(count) == [p + 1000 * ((p - 1) ÷ 4 + 1) + 100000 * ((p - 1) % 4 + 1) + 10^7 * n for p in 1:n]
    end

    @testset "exact multiple of the workgroup size" begin
        A = AT(zeros(Int, dims))
        count = AT(zeros(Int, 8))
        indexmap_mark!(backend, 4)(A, count; ndrange = AT(indices[1:8]))
        synchronize(backend)
        @test sum(Array(A)) == 8
        @test Array(count) == 1:8
    end

    @testset "empty map" begin
        A = AT(zeros(Int, dims))
        indexmap_mark!(backend, 4)(A, AT(Int[]); ndrange = AT(CartesianIndex{3}[]))
        synchronize(backend)
        @test all(iszero, Array(A))
    end

    @testset "errors" begin
        A = AT(zeros(Int, dims))
        count = AT(zeros(Int, n))
        map = AT(indices)
        @test_throws ErrorException indexmap_mark!(backend)(A, count; ndrange = map)
        @test_throws ErrorException indexmap_mark!(backend, (2, 2))(A, count; ndrange = map)
        @test_throws ErrorException KernelAbstractions.partition(indexmap_mark!(backend, 4, (n,)), map, nothing)
        @test_throws ArgumentError indexmap_mark!(backend, 4)(A, count; ndrange = AT([1, 2, 3]))
    end
    return
end
