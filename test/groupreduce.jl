@kernel cpu=false function groupreduce_1!(y, x, op, neutral)
    i = @index(Global)
    val = i > length(x) ? neutral : x[i]
    res = @groupreduce(op, val)
    i == 1 && (y[1] = res)
end

@kernel cpu=false function groupreduce_2!(y, x, op, neutral, ::Val{groupsize}) where {groupsize}
    i = @index(Global)
    val = i > length(x) ? neutral : x[i]
    res = @groupreduce(op, val, groupsize)
    i == 1 && (y[1] = res)
end

@kernel cpu=false function warp_groupreduce_1!(y, x, op, neutral)
    i = @index(Global)
    val = i > length(x) ? neutral : x[i]
    res = @warp_groupreduce(op, val, neutral)
    i == 1 && (y[1] = res)
end

@kernel cpu=false function warp_groupreduce_2!(y, x, op, neutral, ::Val{groupsize}) where {groupsize}
    i = @index(Global)
    val = i > length(x) ? neutral : x[i]
    res = @warp_groupreduce(op, val, neutral, groupsize)
    i == 1 && (y[1] = res)
end

function groupreduce_testsuite(backend, AT)
    # TODO should be a better way of querying max groupsize
    groupsizes = "$backend" == "oneAPIBackend" ?
        (256,) :
        (256, 512, 1024)

    @testset "@groupreduce" begin
        @testset "T=$T, n=$n" for T in (Float16, Float32, Int16, Int32, Int64), n in groupsizes
            x = AT(ones(T, n))
            y = AT(zeros(T, 1))
            neutral = zero(T)
            op = +

            groupreduce_1!(backend(), n)(y, x, op, neutral; ndrange = n)
            @test Array(y)[1] == n

            for groupsize in (64, 128)
                groupreduce_2!(backend())(y, x, op, neutral, Val(groupsize); ndrange = n)
                @test Array(y)[1] == groupsize
            end
        end
    end

    if KernelAbstractions.supports_warp_reduction(backend())
        @testset "@warp_groupreduce" begin
            @testset "T=$T, n=$n" for T in (Float16, Float32, Int16, Int32, Int64), n in groupsizes
                x = AT(ones(T, n))
                y = AT(zeros(T, 1))
                neutral = zero(T)
                op = +

                warp_groupreduce_1!(backend(), n)(y, x, op, neutral; ndrange = n)
                @test Array(y)[1] == n

                for groupsize in (64, 128)
                    warp_groupreduce_2!(backend())(y, x, op, neutral, Val(groupsize); ndrange = n)
                    @test Array(y)[1] == groupsize
                end
            end
        end
    end
end
