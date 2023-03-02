using KernelAbstractions, Test
import KernelAbstractions: allocate, copyto!

function copyto_testsuite(Backend)
    M = 1024
    backend = Backend()

    A = rand!(allocate(backend, Float64, M))
    B = rand!(allocate(backend, Float64, M))

    a = Array{Float64}(undef, M)
    copyto!(backend, a, B)
    copyto!(backend, A, a)
    synchronize(backend)

    @test isapprox(a, Array(A))
    @test isapprox(a, Array(B))
end
