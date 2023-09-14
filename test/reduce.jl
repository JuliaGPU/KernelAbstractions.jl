using KernelAbstractions, Test




@kernel function reduce(a, b, op, neutral)
    idx_in_group = @index(Local)
    
    val = a[idx_in_group]

    val = @groupreduce(op, val, netral)

    b[1] = val 
end

function(backend, ArrayT)
    @testset "groupreduce one group" begin
        @testset for op in (+,*,max,min)
            @testset for type in (Int32, Float32, Float64)
                @test test_1group_groupreduce(backend, ArrayT ,op, type, op(neutral))
            end
        end
    end
end

function test_1group_groupreduce(backend,ArrayT, op, type, neutral)
    a = rand(type, 32)
    b = ArrayT(a)  

    c = similar(b,1)
    reduce(a, c, op, neutral)

    expected = mapreduce(x->x^2, +, a)
    actual = c[1]
    return expected = actual
end



