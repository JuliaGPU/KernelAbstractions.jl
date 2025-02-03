@kernel cpu=false function groupreduce_1!(y, x, op, neutral)
    i = @index(Global)
    val = i > length(x) ? neutral : x[i]
    res = @groupreduce(op, val, neutral)
    i == 1 && (y[1] = res)
end

@kernel cpu=false function groupreduce_2!(y, x, op, neutral, ::Val{groupsize}) where {groupsize}
    i = @index(Global)
    val = i > length(x) ? neutral : x[i]
    res = @groupreduce(op, val, neutral, groupsize)
    i == 1 && (y[1] = res)
end

function groupreduce_testsuite(backend, AT)
    # TODO should be a better way of querying max groupsize
    groupsizes = "$backend" == "oneAPIBackend" ?
        (256,) :
        (256, 512, 1024)
    @testset "@groupreduce" begin
        @testset "T=$T, n=$n" for T in (Float16, Float32, Float64, Int16, Int32, Int64), n in groupsizes
            x = AT(ones(T, n))
            y = AT(zeros(T, 1))

            groupreduce_1!(backend(), n)(y, x, +, zero(T); ndrange = n)
            @test Array(y)[1] == n

            groupreduce_2!(backend())(y, x, +, zero(T), Val(128); ndrange = n)
            @test Array(y)[1] == 128

            groupreduce_2!(backend())(y, x, +, zero(T), Val(64); ndrange = n)
            @test Array(y)[1] == 64
        end
    end
end
