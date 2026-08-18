# These are the KernelInterface tests to be run by implementing backends
module Testsuite

import ..KernelInterface as KI
using ..Test

# We can't add test-dependencies withouth breaking backend packages
const Pkg = Base.require(
    Base.PkgId(
        Base.UUID("44cfe95a-1eb2-52ea-b672-e2afdf69b78f"), "Pkg",
    ),
)

macro conditional_testset(name, skip_tests, expr)
    return esc(
        quote
            @testset $name begin
                if $name ∉ $skip_tests
                    $expr
                else
                    @test_skip false
                end
            end
        end,
    )
end


include("interface.jl")

function testsuite(backend, backend_str, backend_mod, AT, DAT; skip_tests = Set{String}())
    @conditional_testset "Interface" skip_tests begin
        interface_testsuite(backend, AT)
    end

    return
end

end
