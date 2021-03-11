using KernelAbstractions, Test

function asynccopy_testsuite(backend, ArrayT)
    M = 1024

    A = ArrayT(rand(Float64, M))
    B = ArrayT(rand(Float64, M))

    a = Array{Float64}(undef, M)
    event = async_copy!(backend(), a, B, dependencies=Event(CPU()))
    event = async_copy!(backend(), A, a, dependencies=event)
    wait(event)

    @test isapprox(a, Array(A))
    @test isapprox(a, Array(B))
end
