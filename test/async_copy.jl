using KernelAbstractions, Test

function asynccopy_testsuite(backend, backend_str, ArrayT)
    M = 1024

    T = backend_str == "Metal" ? Float32 : Float64

    A = ArrayT(rand(T, M))
    B = ArrayT(rand(T, M))

    a = Array{T}(undef, M)
    event = async_copy!(backend(), a, B, dependencies=Event(CPU()))
    event = async_copy!(backend(), A, a, dependencies=event)
    wait(event)

    @test isapprox(a, Array(A))
    @test isapprox(a, Array(B))
end
