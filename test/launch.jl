using Adapt
using KernelAbstractions
using KernelAbstractions.NDIteration
import KernelAbstractions.KernelInterface as KI
using Test

@kernel function launch_indices!(GL, GC, BL, BC, LL, LC, WS, lo)
    I = @index(Global, NTuple)
    J = I .- lo .+ 1
    @inbounds begin
        GL[J...] = @index(Global, Linear)
        GC[J...] = @index(Global, Cartesian)
        BL[J...] = @index(Group, Linear)
        BC[J...] = @index(Group, Cartesian)
        LL[J...] = @index(Local, Linear)
        LC[J...] = @index(Local, Cartesian)
        WS[J...] = CartesianIndex(@groupsize())
    end
end

# `unsafe_indices` kernels compute global indices from the group and local ones
@kernel unsafe_indices = true function launch_unsafe!(A)
    g = @index(Group, NTuple)
    l = @index(Local, NTuple)
    I = (g .- 1) .* @groupsize() .+ l
    if all(I .<= size(A))
        @inbounds A[I...] = LinearIndices(A)[I...]
    end
end

# padding lanes of partial workgroups have to reach the barrier too
@kernel function launch_sync!(A)
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    N = @uniform prod(@groupsize())
    lmem = @localmem Int (N,)
    @inbounds lmem[i] = I
    @synchronize
    @inbounds A[I] = lmem[i]
end

# more arguments than Julia splats efficiently (32)
const MANY_ARGS = [Symbol(:x, i) for i in 1:40]
@eval @kernel function launch_many!(A, $(MANY_ARGS...))
    I = @index(Global, Linear)
    @inbounds A[I] = $(foldl((a, b) -> :($a + $b), MANY_ARGS))
end

# A custom iteration space: one work-item per index in a list, whose linear index is its
# position in the list (as Oceananigans launches kernels over the active cells of a grid)
struct IndexList{V <: AbstractVector}
    indices::V
end
Adapt.@adapt_structure IndexList

# the `ndrange` of the context: every index but `UNLISTED` is part of the space
struct ListedIndices
    length::Int
end
const UNLISTED = CartesianIndex(typemin(Int), typemin(Int))
Base.in(I::CartesianIndex{2}, ::ListedIndices) = I != UNLISTED

const ListNDRange = NDRange{1, <:Any, <:Any, <:Any, <:Any, <:IndexList}

function KernelAbstractions.partition(kernel::KernelAbstractions.Kernel, list::IndexList, workgroupsize)
    static_workgroupsize = KernelAbstractions.workgroupsize(kernel)
    items = NDIteration.get(static_workgroupsize)
    blocks, _, dynamic = NDIteration.partition((length(list.indices),), items)
    return NDRange{1, DynamicSize, static_workgroupsize}(CartesianIndices(blocks), nothing, list), dynamic
end
KernelAbstractions.cartesian(list::IndexList) = ListedIndices(length(list.indices))
KernelAbstractions.cartesian(r::ListedIndices) = r

@inline list_position(r::ListNDRange, g::CartesianIndex{1}, i::CartesianIndex{1}) =
    (g[1] - 1) * length(workitems(r)) + i[1]
@inline function NDIteration.expand(r::ListNDRange, g::CartesianIndex{1}, i::CartesianIndex{1})
    p = list_position(r, g, i)
    return p <= length(r.mapping.indices) ? (@inbounds r.mapping.indices[p]) : UNLISTED
end
@inline NDIteration.linear_index(r::ListNDRange, ::ListedIndices, g::CartesianIndex{1}, i::CartesianIndex{1}) =
    list_position(r, g, i)

@kernel function launch_listed!(visits, position)
    I = @index(Global, Cartesian)
    @inbounds begin
        visits[I] += 1
        position[I] = @index(Global, Linear)
    end
end

default_launcher(kernel, args...; ndrange, workgroupsize = nothing) =
    kernel(args...; ndrange, workgroupsize)

# Check every `@index` flavour against the layout KernelAbstractions defines: groups and
# work-items are numbered column-major, and the global index is `(g-1)*groupsize + l`.
function check_indices(launcher, backend, AT, kernel, ndrange; workgroupsize = nothing)
    ranges = map(r -> r isa Integer ? (1:r) : r, ndrange)
    N = length(ranges)
    ext = map(length, ranges)
    lo = map(first, ranges)
    arrays = map((Int, CartesianIndex{N}, Int, CartesianIndex{N}, Int, CartesianIndex{N}, CartesianIndex{N})) do T
        AT(zeros(T, ext))
    end
    launcher(kernel, arrays..., lo; ndrange, workgroupsize)
    synchronize(backend)
    GL, GC, BL, BC, LL, LC, WS = map(Array, arrays)

    wgs = Tuple(first(WS))
    all(==(CartesianIndex(wgs)), WS) || return false
    groups = cld.(ext, wgs)
    for J in CartesianIndices(ext)
        g = cld.(J.I, wgs)
        l = J.I .- (g .- 1) .* wgs
        GL[J] == LinearIndices(ext)[J] || return false
        GC[J] == CartesianIndex(J.I .+ lo .- 1) || return false
        BC[J] == CartesianIndex(g) || return false
        BL[J] == LinearIndices(groups)[g...] || return false
        LC[J] == CartesianIndex(l) || return false
        LL[J] == LinearIndices(wgs)[l...] || return false
    end
    return true
end

function launch_testsuite(backend, AT; launcher = default_launcher, skip_tests = Set{String}())
    @testset "index layout" begin
        shapes = Tuple[(), (7,), (37,), (5, 7), (33, 3), (3, 5, 7), (2, 3, 4, 5)]
        @testset "$shape, workgroupsize=$wgs" for shape in shapes,
                wgs in (nothing, 4, (2, 3), (4, 1, 2))
            wgs !== nothing && length(wgs) > length(shape) && continue
            @test check_indices(
                launcher, backend(), AT, launch_indices!(backend()), shape;
                workgroupsize = wgs
            )
        end

        @testset "static workgroupsize" begin
            @test check_indices(launcher, backend(), AT, launch_indices!(backend(), (4, 2)), (9, 5))
            @test check_indices(launcher, backend(), AT, launch_indices!(backend(), 8), (9, 5, 3))
        end

        @testset "static ndrange" begin
            @test check_indices(launcher, backend(), AT, launch_indices!(backend(), (4, 2), (9, 5)), (9, 5))
            @test check_indices(launcher, backend(), AT, launch_indices!(backend(), (4, 2), (0:8, -2:2)), (0:8, -2:2))

            # with a tuned workgroup size, over more work-items than fit a workgroup
            n = KI.max_work_group_size(backend()) + 1
            kernel = launch_indices!(backend(), KernelAbstractions.DynamicSize(), KernelAbstractions.StaticSize((n, 3)))
            @test check_indices(launcher, backend(), AT, kernel, (n, 3))
            # a given ndrange has to agree with the static one
            @test_throws ErrorException check_indices(launcher, backend(), AT, kernel, (n, 2))
        end

        @testset "offsets" begin
            @test check_indices(launcher, backend(), AT, launch_indices!(backend()), (-3:4,))
            @test check_indices(launcher, backend(), AT, launch_indices!(backend()), (-3:4, 2:11); workgroupsize = (3, 3))
            @test check_indices(launcher, backend(), AT, launch_indices!(backend()), (0:4, 3, 2:3))
        end
    end

    @testset "empty ndrange" begin
        for shape in ((0,), (0, 5), (5, 0), (3, 4, 0), (2, 0, 2, 2))
            A = AT(zeros(Int, max.(shape, 1)))
            launcher(launch_unsafe!(backend()), A; ndrange = shape)
            synchronize(backend())
            @test all(iszero, Array(A))
        end
    end

    @testset "unsafe_indices" begin
        for (shape, wgs) in (((37,), 8), ((33, 7), (8, 4)), ((9, 5, 3), (4, 2, 2)), ((9, 5), nothing))
            A = AT(zeros(Int, shape))
            launcher(launch_unsafe!(backend()), A; ndrange = shape, workgroupsize = wgs)
            synchronize(backend())
            @test Array(A) == LinearIndices(A)
        end
    end

    # back ends that limit the number of kernel arguments (Metal: 31 buffers) can skip this
    @conditional_testset "many arguments" skip_tests begin
        A = AT(zeros(Int, 5))
        launcher(launch_many!(backend()), A, 1:40...; ndrange = length(A))
        synchronize(backend())
        @test all(==(sum(1:40)), Array(A))
    end

    @testset "custom iteration space" begin
        shape = (7, 5)
        # 18 indices: the last workgroup of 4 work-items is partial
        listed = [I for I in CartesianIndices(shape) if isodd(sum(Tuple(I)))]
        visits = AT(zeros(Int, shape))
        position = AT(zeros(Int, shape))
        launcher(launch_listed!(backend(), 4), visits, position; ndrange = IndexList(AT(listed)))
        synchronize(backend())
        visits, position = Array(visits), Array(position)
        @test all(I -> visits[I] == (I in listed), CartesianIndices(shape))
        @test all(p -> position[listed[p]] == p, eachindex(listed))
    end

    @testset "synchronize with padding lanes" begin
        for (shape, wgs) in (((37,), (8,)), ((7, 6), (4, 4)), ((5, 3, 3), (2, 2, 2)))
            A = AT(zeros(Int, shape))
            launcher(launch_sync!(backend(), wgs), A; ndrange = shape)
            synchronize(backend())
            @test Array(A) == LinearIndices(A)
        end
    end
    return
end

function select_launch_testsuite()
    # the limits of a CUDA GPU
    max_items = 1024
    max_dims = (1024, 1024, 64)
    max_groups = (Int(typemax(Int32)), 65535, 65535)
    select(extent, groupsize = nothing; max_dims = max_dims, max_groups = max_groups) =
        KernelAbstractions.select_launch(extent, groupsize, max_items, max_dims, max_groups)
    LinearLaunch = KernelAbstractions.LinearLaunch
    NDLaunch = KernelAbstractions.NDLaunch

    @testset "selection" begin
        # as many dimensions as the grid has are launched as such
        @test select((1000,)) === NDLaunch{Int32}()
        @test select((100, 100)) === NDLaunch{Int32}()
        @test select((10, 10, 10), (4, 4, 4)) === NDLaunch{Int32}()
        @test select(()) === NDLaunch{Int32}()
        @test select((2, 3, 4, 5)) === LinearLaunch{Int32}()
        @test select((100, 100); max_dims = (1024, 1024), max_groups = (65535, 65535)) ===
            NDLaunch{Int32}()
        @test select((10, 10, 10); max_dims = (1024, 1024), max_groups = (65535, 65535)) ===
            LinearLaunch{Int32}()

        # unless that exceeds the per-dimension limits
        @test select((100, 100_000)) === LinearLaunch{Int32}()
        @test select((1, 1, 5000), (1, 1, 128)) === LinearLaunch{Int32}()
        @test select((10,), (2048,)) === LinearLaunch{Int32}()
        @test select((64, 64), (64, 64)) === LinearLaunch{Int32}()
        # tuning respects the per-dimension limits
        @test select((1, 1, 5000)) === NDLaunch{Int32}()

        # indices are computed in Int32 if the padded iteration space fits
        @test select((1024, 1024, 1024)) === NDLaunch{Int32}()
        @test select((1025, 1024, 1024)) === NDLaunch{Int}()
        @test select((2^31 - 1024,)) === NDLaunch{Int32}()
        @test select((2^31 - 1023,)) === NDLaunch{Int}()
        @test select((2^31 - 1,), (1,)) === NDLaunch{Int32}()
        @test select((2^31 - 1,), (2,)) === NDLaunch{Int}()
        @test select((2^16, 2^16, 2), (256,)) === LinearLaunch{Int}()
        # ... and don't have to be representable at all
        @test_throws ArgumentError select((2^40, 2^40))
        @test_throws ArgumentError select((typemax(Int),))
        @test_throws ArgumentError select((typemax(Int),), (2,))
        # including when empty
        @test select((2^40, 2^40, 0)) === LinearLaunch{Int32}()
        @test select((2^20, 2^12, 0), (1024,)) === NDLaunch{Int32}()
    end

    # The index type is chosen before the workgroup size is tuned, so the padding that the
    # tuned workgroup introduces has to be bounded for every thread count.
    @testset "tuned padding bound" begin
        for extent in (
                (5,), (1000,), (3, 7), (33, 1000), (1, 1, 5000), (7, 9, 11),
                (1500, 3, 2), (2, 3, 4, 5), (1, 1, 1, 3000), (0, 7),
            )
            for limits in ((), max_dims)
                bound = KernelAbstractions.tuned_padded(extent, max_items, limits)
                @test all(1:max_items) do threads
                    wgs = KI.threads_to_workgroupsize(threads, extent, limits)
                    padded = cld.(extent, wgs) .* wgs
                    all(padded .<= bound)
                end
            end
        end
    end
    return
end
