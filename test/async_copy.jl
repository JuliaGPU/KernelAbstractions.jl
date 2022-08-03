using KernelAbstractions, Test

function asynccopy_testsuite(backend, ArrayT)
    M = 1024

    A = ArrayT(rand(Float64, M))
    B = ArrayT(rand(Float64, M))

    a = Array{Float64}(undef, M)
    async_copy!(backend(), a, B)
    async_copy!(backend(), A, a)
    synchronize(backend())

    @test isapprox(a, Array(A))
    @test isapprox(a, Array(B))
end
