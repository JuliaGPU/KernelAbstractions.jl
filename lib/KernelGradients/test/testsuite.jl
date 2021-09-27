module GradientsTestsuite

using ..KernelAbstractions
using ..KernelGradients
using ..Enzyme
using ..Test

include("matmul.jl")

function testsuite(backend, backend_str, backend_mod, AT, DAT)
    @testset "Matmul" begin
        matmul_testsuite(backend, AT)
    end
end

end
