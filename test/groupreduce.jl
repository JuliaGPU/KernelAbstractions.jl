@kernel function groupreduce_1!(y, x, op, neutral, algo)
    i = @index(Global)
    val = i > length(x) ? neutral : x[i]
    res = @groupreduce(op, val, neutral, algo)
    i == 1 && (y[1] = res)
end

@kernel function groupreduce_2!(y, x, op, neutral, algo, ::Val{groupsize}) where {groupsize}
    i = @index(Global)
    val = i > length(x) ? neutral : x[i]
    res = @groupreduce(op, val, neutral, algo, groupsize)
    i == 1 && (y[1] = res)
end

function groupreduce_testsuite(backend, AT)
    # TODO should be better way of querying max groupsize
    groupsizes = "$backend" == "oneAPIBackend" ?
        (256,) :
        (256, 512, 1024)
    @testset "@groupreduce" begin
        @testset "thread reduction T=$T, n=$n" for T in (Float16, Float32, Int32, Int64), n in groupsizes
            x = AT(ones(T, n))
            y = AT(zeros(T, 1))

            groupreduce_1!(backend(), n)(y, x, +, zero(T), Reduction.thread; ndrange = n)
            @test Array(y)[1] == n

            groupreduce_2!(backend())(y, x, +, zero(T), Reduction.thread, Val(128); ndrange = n)
            @test Array(y)[1] == 128

            groupreduce_2!(backend())(y, x, +, zero(T), Reduction.thread, Val(64); ndrange = n)
            @test Array(y)[1] == 64
        end

        warp_reduction = KernelAbstractions.supports_warp_reduction(backend())
        if warp_reduction
            @testset "warp reduction T=$T, n=$n" for T in (Float16, Float32, Int32, Int64), n in groupsizes
                x = AT(ones(T, n))
                y = AT(zeros(T, 1))
                groupreduce_1!(backend(), n)(y, x, +, zero(T), Reduction.warp; ndrange = n)
                @test Array(y)[1] == n

                groupreduce_2!(backend())(y, x, +, zero(T), Reduction.warp, Val(128); ndrange = n)
                @test Array(y)[1] == 128

                groupreduce_2!(backend())(y, x, +, zero(T), Reduction.warp, Val(64); ndrange = n)
                @test Array(y)[1] == 64
            end
        end
    end
end
