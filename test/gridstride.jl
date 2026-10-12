using KernelAbstractions
import KernelAbstractions.KernelInterface as KI
using Test

@kernel function gridstride_indices!(glin, gcart, group, loc, ngroups)
    I = @index(Global, Linear)
    C = @index(Global, Cartesian)
    @inbounds begin
        glin[C] = I
        gcart[C] = C
        group[C] = @index(Group, Linear)
        loc[C] = @index(Local, Linear)
        ngroups[C] = KI.get_num_groups().x
    end
end

# `return` skips the rest of the body for one index, not the remaining groups
@kernel unsafe_indices = true function gridstride_early_return!(A, n)
    i = @index(Global, Linear)
    i > n && return
    @inbounds A[i] = i
end

# reverses every workgroup's share of `A` through local memory, which races without a
# barrier between the groups a hardware group strides over: the first half of the group
# dawdles before it reads the slots that the second half overwrites in its next group.
# (The address of the read depends on the delay loop, so that it can't be hoisted above it.)
@kernel function gridstride_reverse!(B, @Const(A))
    i = @index(Global, Linear)
    l = @index(Local, Linear)
    N = @uniform @groupsize()[1]
    tile = @localmem eltype(A) (N,)
    @inbounds tile[l] = A[i]
    @synchronize
    acc = 0.0f0
    if l <= N ÷ 2
        for _ in 1:2000
            acc = muladd(acc, 0.999f0, 1.0f-3)
        end
    end
    @inbounds B[i] = tile[N - l + 1 - Int(acc < 0)]
end

@kernel function gridstride_nosync!(A)
    A[@index(Global)] = 1
end

function gridstride_reference(backend, AT, ndrange, wgs)
    sz = KernelAbstractions.NDIteration.extents(ndrange)
    arrays = (
        KernelAbstractions.zeros(backend, Int, sz...),
        AT(fill(CartesianIndex(ntuple(_ -> 0, length(sz))), sz)),
        KernelAbstractions.zeros(backend, Int, sz...),
        KernelAbstractions.zeros(backend, Int, sz...),
        KernelAbstractions.zeros(backend, Int, sz...),
    )
    return arrays
end

function gridstride_testsuite(Backend, AT)
    backend = Backend()

    @testset "indices agree with a launch without a stride" begin
        for (ndrange, wgs) in (
                (1000, 64), ((37, 23), (8, 4)), ((3, 4, 5, 6), (2, 2)), ((5, 7), nothing),
            )
            sz = KernelAbstractions.NDIteration.extents(ndrange)
            nblocks = prod(first(KernelAbstractions.NDIteration.partition(sz, something(wgs, sz))))
            ref = gridstride_reference(backend, AT, ndrange, wgs)
            gridstride_indices!(backend)(ref...; ndrange, workgroupsize = wgs)
            for gs in (1, 3, true, 10^6)
                out = gridstride_reference(backend, AT, ndrange, wgs)
                gridstride_indices!(backend)(out...; ndrange, workgroupsize = wgs, gridstride = gs)
                synchronize(backend)
                # the hardware groups are the only difference
                for k in 1:4
                    @test Array(out[k]) == Array(ref[k])
                end
                ngroups = Array(out[5])
                @test allequal(ngroups)
                if gs isa Int && wgs !== nothing
                    @test first(ngroups) == min(gs, nblocks)
                elseif wgs !== nothing
                    @test first(ngroups) <= nblocks
                end
            end
        end
    end

    @testset "static sizes" begin
        out = gridstride_reference(backend, AT, (37, 23), (8, 4))
        ref = gridstride_reference(backend, AT, (37, 23), (8, 4))
        gridstride_indices!(backend, (8, 4), (37, 23))(ref...)
        gridstride_indices!(backend, (8, 4), (37, 23))(out...; gridstride = 2)
        synchronize(backend)
        for k in 1:4
            @test Array(out[k]) == Array(ref[k])
        end
        @test all(==(2), Array(out[5]))
    end

    @testset "offset ndrange" begin
        A = KernelAbstractions.zeros(backend, Int, 9, 5)
        B = KernelAbstractions.zeros(backend, Int, 9, 5)
        foreach_index(backend, CartesianIndices((2:9, 1:5)); workgroupsize = (4, 2), gridstride = 3) do I
            @inbounds A[I] = I[1] + 10 * I[2]
        end
        foreach_index(backend, CartesianIndices((2:9, 1:5))) do I
            @inbounds B[I] = I[1] + 10 * I[2]
        end
        synchronize(backend)
        @test Array(A) == Array(B)
    end

    @testset "early return" begin
        A = KernelAbstractions.zeros(backend, Int, 1024)
        gridstride_early_return!(backend, 32)(A, 1000; ndrange = 1024, gridstride = 2)
        synchronize(backend)
        @test Array(A) == [i <= 1000 ? i : 0 for i in 1:1024]
    end

    @testset "local memory and @synchronize" begin
        N, groups = 256, 1024
        A = AT(collect(1:(N * groups)))
        B = KernelAbstractions.zeros(backend, Int, N * groups)
        for gridstride in (2, true)
            gridstride_reverse!(backend, N)(B, A; ndrange = length(A), gridstride)
            synchronize(backend)
            @test Array(B) == reduce(vcat, reverse.(Iterators.partition(1:(N * groups), N)))
        end
    end

    @testset "SyncQuery" begin
        A = KernelAbstractions.zeros(backend, Int, 4)
        dA = KI.argconvert(backend, A)
        @test gpu_gridstride_reverse!(KernelAbstractions.SyncQuery(), dA, dA) === Val(true)
        @test gpu_gridstride_nosync!(KernelAbstractions.SyncQuery(), dA) === Val(false)
        @test gpu_gridstride_early_return!(KernelAbstractions.SyncQuery(), dA, 1) === Val(true)
    end

    @testset "foreach_index" begin
        A = KernelAbstractions.zeros(backend, Float32, 10_000)
        foreach_index(A; gridstride = true) do i
            @inbounds A[i] = i
        end
        synchronize(backend)
        @test Array(A) == 1:10_000
    end

    @testset "invalid gridstride" begin
        A = KernelAbstractions.zeros(backend, Int, 4)
        @test_throws ArgumentError gridstride_nosync!(backend)(A; ndrange = 4, gridstride = 0)
        @test_throws ArgumentError gridstride_nosync!(backend)(A; ndrange = 4, gridstride = :auto)
    end
    return
end
