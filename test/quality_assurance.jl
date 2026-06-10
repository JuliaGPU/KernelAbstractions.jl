using Aqua

function quality_assurance_testsuite()
    @testset "Aqua" begin
        Aqua.test_all(KernelAbstractions)
    end
    return nothing
end
