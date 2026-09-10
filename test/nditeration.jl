using KernelAbstractions
using KernelAbstractions.NDIteration
using Test

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
