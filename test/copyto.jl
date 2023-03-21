using KernelAbstractions, Test
import KernelAbstractions: allocate, copyto!
using Random

function copyto_testsuite(Backend)
    M = 1024
    backend = Backend()
    ET = KernelAbstractions.is_float64_suppported(backend) ? Float64 : Float32

    A = rand!(allocate(backend, ET, M))
    B = rand!(allocate(backend, ET, M))

    a = Array{ET}(undef, M)
    copyto!(backend, a, B)
    copyto!(backend, A, a)
    synchronize(backend)

    @test isapprox(a, Array(A))
    @test isapprox(a, Array(B))
end
