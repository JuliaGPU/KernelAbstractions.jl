using Aqua

function quality_assurance_testsuite()
    @testset "Aqua" begin
        Aqua.test_all(
            KernelAbstractions;
            stale_deps = (; ignore = [:Enzyme, :EnzymeCore]),
        )
    end
    return nothing
end
