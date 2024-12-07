using KernelAbstractions, Test
import KernelAbstractions: allocate, copyto!
using Random

function copyto_testsuite(Backend, ArrayT)
    M = 1024
    backend = Backend()
    ET = KernelAbstractions.supports_float64(backend) ? Float64 : Float32

    A = ArrayT(rand(ET, M))
    B = ArrayT(rand(ET, M))

    a = Array{ET}(undef, M)
    copyto!(backend, a, B)
    copyto!(backend, A, a)
    synchronize(backend)

    @test isapprox(a, Array(A))
    @test isapprox(a, Array(B))
    return
end
