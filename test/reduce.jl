using KernelAbstractions, Test

@kernel function reduce(a, b, op, neutral)
    I = @index(Global)
    gI = @index(Group, Linear)
    val = a[I]

    val = @groupreduce(op, val, neutral)

    b[gI] = val 
end

function reduce_testset(backend, ArrayT)
    @testset "groupreduce one group" begin
        @testset for op in (+, *, max, min)
            @testset for type in (Int32, Float32, Float64)
                @test test_groupreduce(backend, ArrayT, op, type, op(neutral), 8)
                @test test_groupreduce(backend, ArrayT, op, type, op(neutral), 16)
                @test test_groupreduce(backend, ArrayT, op, type, op(neutral), 32)
                @test test_groupreduce(backend, ArrayT, op, type, op(neutral), 64)
            end
        end
    end
end

function test_groupreduce(backend, ArrayT, op, type, neutral, N)
    a = rand(type, N)
    b = ArrayT(a)

    gsz = 64
    ngroups = ceil(N/gsz)
    c = similar(b, ngroups)
    kernel = reduce(backend, (gsz,))
    kernel(a, c, op, neutral)

    expected = mapreduce(x->x^2, +, a)
    actual = c[1]
    return expected = actual
end



