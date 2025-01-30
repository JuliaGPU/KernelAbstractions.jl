@kernel function groupreduce_thread_1!(y, x, op, neutral)
    i = @index(Global)
    val = i > length(x) ? neutral : x[i]
    res = KernelAbstractions.@groupreduce(:thread, op, val, neutral)
    i == 1 && (y[1] = res)
end

@kernel function groupreduce_thread_2!(y, x, op, neutral, ::Val{groupsize}) where {groupsize}
    i = @index(Global)
    val = i > length(x) ? neutral : x[i]
    res = KernelAbstractions.@groupreduce(:thread, op, val, neutral, groupsize)
    i == 1 && (y[1] = res)
end

@kernel function groupreduce_warp_1!(y, x, op, neutral)
    i = @index(Global)
    val = i > length(x) ? neutral : x[i]
    res = KernelAbstractions.@groupreduce(:warp, op, val, neutral)
    i == 1 && (y[1] = res)
end

@kernel function groupreduce_warp_2!(y, x, op, neutral, ::Val{groupsize}) where {groupsize}
    i = @index(Global)
    val = i > length(x) ? neutral : x[i]
    res = KernelAbstractions.@groupreduce(:warp, op, val, neutral, groupsize)
    i == 1 && (y[1] = res)
end

function groupreduce_testsuite(backend, AT)
    @testset "@groupreduce" begin
        @testset ":thread T=$T, n=$n" for T in (Float16, Float32, Int32, Int64), n in (256, 512, 1024)
            x = AT(ones(T, n))
            y = AT(zeros(T, 1))

            groupreduce_thread_1!(backend(), n)(y, x, +, zero(T); ndrange=n)
            @test Array(y)[1] == n

            groupreduce_thread_2!(backend())(y, x, +, zero(T), Val(128); ndrange=n)
            @test Array(y)[1] == 128

            groupreduce_thread_2!(backend())(y, x, +, zero(T), Val(64); ndrange=n)
            @test Array(y)[1] == 64
        end

        @testset ":warp T=$T, n=$n" for T in (Float16, Float32, Int32, Int64), n in (256, 512, 1024)
            x = AT(ones(T, n))
            y = AT(zeros(T, 1))
            groupreduce_warp_1!(backend(), n)(y, x, +, zero(T); ndrange=n)
            @test Array(y)[1] == n

            groupreduce_warp_2!(backend())(y, x, +, zero(T), Val(128); ndrange=n)
            @test Array(y)[1] == 128

            groupreduce_warp_2!(backend())(y, x, +, zero(T), Val(64); ndrange=n)
            @test Array(y)[1] == 64
        end
    end
end
