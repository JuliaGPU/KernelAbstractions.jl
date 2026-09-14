using Adapt
using KernelAbstractions
using KernelAbstractions.NDIteration
using Test

# A mapping holding an array, like a list of indices to iterate over
struct ArrayMapping{A}
    array::A
end
Adapt.@adapt_structure ArrayMapping

function nditeration_testsuite()
    @testset "iteration" begin
        let ndrange = NDRange{2, DynamicSize, DynamicSize}(CartesianIndices((256, 256)), CartesianIndices((32, 32)))
            @test length(ndrange) == 256 * 256
            @test all(p -> p[1] == p[2], zip(ndrange, CartesianIndices((256, 256))))
            @test ndims(ndrange) == 2
        end
        let ndrange = NDRange{2, StaticSize{(256, 256)}, DynamicSize}(nothing, CartesianIndices((32, 32)))
            @test length(ndrange) == 256 * 256
            @test all(p -> p[1] == p[2], zip(ndrange, CartesianIndices((256, 256))))
            @test ndims(ndrange) == 2
        end
    end

    @testset "offsets" begin
        @test NDIteration.get(StaticSize((1:4, 0:9))) == (1:4, 0:9)
        @test NDIteration.get(StaticSize(CartesianIndices((3, 0:9)))) == (3, 0:9)
        @test length(StaticSize((1:4, 0:9))) == 40
        @test extents((1:4, 0:9, 7)) == (4, 10, 7)
        @test extents(CartesianIndices((3, 0:9))) == (3, 10)
        @test extents(0:9) == (10,)
        @test offsets((1:4, 0:9, 7)) == (0, -1, 0)

        let ndrange = NDRange{2, StaticSize{(4, 4)}, StaticSize{(8, 8)}}(nothing, nothing, StaticOffset{(-8, 3)}())
            @test offsets(ndrange) == (-8, 3)
            @test expand(ndrange, CartesianIndex(1, 1), CartesianIndex(1, 1)) == CartesianIndex(-7, 4)
            @test expand(ndrange, CartesianIndex(4, 4), CartesianIndex(8, 8)) == CartesianIndex(24, 35)
        end
        let ndrange = NDRange{2, DynamicSize, DynamicSize}(CartesianIndices((4, 4)), CartesianIndices((8, 8)), DynamicOffset((-8, 3)))
            @test offsets(ndrange) == (-8, 3)
            @test expand(ndrange, 1, 1) == CartesianIndex(-7, 4)
            @test expand(ndrange, 16, 64) == CartesianIndex(24, 35)
        end
        let ndrange = NDRange{2, DynamicSize, DynamicSize}(CartesianIndices((4, 4)), CartesianIndices((8, 8)))
            @test offsets(ndrange) == (0, 0)
            @test ndrange.mapping === nothing
        end

        let ci = CartesianIndices((-3:4, 2:11))
            @test linear_index(ci, CartesianIndex(-3, 2)) == 1
            @test linear_index(ci, CartesianIndex(4, 2)) == 8
            @test linear_index(ci, CartesianIndex(4, 11)) == 80
        end
    end

    @testset "adapt" begin
        mapping = ArrayMapping([1, 2, 3])
        ndrange = NDRange{1, DynamicSize, StaticSize{(4,)}}(CartesianIndices((2,)), nothing, mapping)
        adapted = adapt(Array{Float32}, ndrange)
        @test adapted isa NDRange{1, DynamicSize, StaticSize{(4,)}}
        @test blocks(adapted) == blocks(ndrange)
        @test adapted.mapping.array isa Vector{Float32}
        @test adapted.mapping.array == [1, 2, 3]

        # a mapping without device data is left alone
        offset = NDRange{2, DynamicSize, DynamicSize}(CartesianIndices((4, 4)), CartesianIndices((8, 8)), DynamicOffset((-8, 3)))
        @test adapt(Array{Float32}, offset).mapping === offset.mapping

        # GPU-style context: the index is implicit
        ctx = KernelAbstractions.CompilerMetadata{DynamicSize, DynamicCheck}(CartesianIndices((8,)), ndrange)
        actx = adapt(Array{Float32}, ctx)
        @test actx isa KernelAbstractions.CompilerMetadata{DynamicSize, DynamicCheck}
        @test KernelAbstractions.__groupindex(actx) === nothing
        @test KernelAbstractions.__ndrange(actx) == CartesianIndices((8,))
        @test KernelAbstractions.__iterspace(actx).mapping.array isa Vector{Float32}

        # CPU-style context: the group index is explicit
        ctx = KernelAbstractions.CompilerMetadata{DynamicSize, DynamicCheck}(CartesianIndex(2), CartesianIndices((8,)), ndrange)
        actx = adapt(Array{Float32}, ctx)
        @test KernelAbstractions.__groupindex(actx) == CartesianIndex(2)
        @test KernelAbstractions.__ndrange(actx) == CartesianIndices((8,))
        @test KernelAbstractions.__iterspace(actx).mapping.array isa Vector{Float32}
    end

    # GPU scenario where we get a linear index into workitems/blocks
    function linear_iteration(ndrange)
        idx = Array{CartesianIndex{2}}(undef, length(blocks(ndrange)) * length(workitems(ndrange)))
        for i in 1:length(blocks(ndrange))
            for j in 1:length(workitems(ndrange))
                I = j + (i - 1) * length(workitems(ndrange))
                idx[I] = expand(ndrange, i, j)
            end
        end
        return idx
    end

    function check(idx, offset, offset_x, offset_y, Dim_x, Dim_y)
        N = Dim_x * Dim_y
        return all(p -> p[1] == p[2], zip(idx[(offset * N .+ 1):N], CartesianIndices(((offset_x * Dim_x .+ 1):Dim_x, (offset_y * Dim_y .+ 1):Dim_y))))
    end

    @testset "linear_iteration" begin
        Dim_x = 32
        Dim_y = 32
        let ndrange = NDRange{2, StaticSize{(4, 4)}, StaticSize{(Dim_x, Dim_y)}}()
            idx = linear_iteration(ndrange)
            for (i, I) in zip(1:length(blocks(ndrange)), blocks(ndrange))
                I = Tuple(I)
                @test check(idx, i - 1, ntuple(i -> I[i] - 1, length(I))..., Dim_x, Dim_y)
            end
            @test ndims(ndrange) == 2
        end
        let ndrange = NDRange{2, DynamicSize, DynamicSize}(CartesianIndices((4, 4)), CartesianIndices((Dim_x, Dim_y)))
            idx = linear_iteration(ndrange)
            for (i, I) in zip(1:length(blocks(ndrange)), blocks(ndrange))
                I = Tuple(I)
                @test check(idx, i - 1, ntuple(i -> I[i] - 1, length(I))..., Dim_x, Dim_y)
            end
            @test ndims(ndrange) == 2
        end

        Dim_x = 32
        Dim_y = 1
        let ndrange = NDRange{2, StaticSize{(4, 4 * 32)}, StaticSize{(Dim_x, Dim_y)}}()
            idx = linear_iteration(ndrange)
            N = length(workitems(ndrange))
            for (i, I) in zip(1:length(blocks(ndrange)), blocks(ndrange))
                I = Tuple(I)
                @test check(idx, i - 1, ntuple(i -> I[i] - 1, length(I))..., Dim_x, Dim_y)
            end
            @test ndims(ndrange) == 2
        end
        let ndrange = NDRange{2, DynamicSize, DynamicSize}(CartesianIndices((4, 4 * 32)), CartesianIndices((Dim_x, Dim_y)))
            idx = linear_iteration(ndrange)
            for (i, I) in zip(1:length(blocks(ndrange)), blocks(ndrange))
                I = Tuple(I)
                @test check(idx, i - 1, ntuple(i -> I[i] - 1, length(I))..., Dim_x, Dim_y)
            end
            @test ndims(ndrange) == 2
        end

        Dim_x = 1
        Dim_y = 32
        let ndrange = NDRange{2, StaticSize{(4 * 32, 4)}, StaticSize{(Dim_x, Dim_y)}}()
            idx = linear_iteration(ndrange)
            N = length(workitems(ndrange))
            for (i, I) in zip(1:length(blocks(ndrange)), blocks(ndrange))
                I = Tuple(I)
                @test check(idx, i - 1, ntuple(i -> I[i] - 1, length(I))..., Dim_x, Dim_y)
            end
            @test ndims(ndrange) == 2
        end
        let ndrange = NDRange{2, DynamicSize, DynamicSize}(CartesianIndices((4 * 32, 4)), CartesianIndices((Dim_x, Dim_y)))
            idx = linear_iteration(ndrange)
            for (i, I) in zip(1:length(blocks(ndrange)), blocks(ndrange))
                I = Tuple(I)
                @test check(idx, i - 1, ntuple(i -> I[i] - 1, length(I))..., Dim_x, Dim_y)
            end
            @test ndims(ndrange) == 2
        end
    end
    return
end
