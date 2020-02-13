using KernelAbstractions
using Test

@testset "Unittests" begin
    include("test.jl")
end

@testset "Localmem" begin
    include("localmem.jl")
end

include("examples.jl")
