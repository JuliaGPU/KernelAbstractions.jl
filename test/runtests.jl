using KernelAbstractions
using CUDAKernels
using Test

@testset "Unittests" begin
    include("test.jl")
end

@testset "Localmem" begin
    include("localmem.jl")
end

@testset "Private" begin
    include("private.jl")
end

@testset "Unroll" begin
    include("unroll.jl")
end

@testset "NDIteration" begin
    include("nditeration.jl")
end

@testset "async_copy!" begin
    include("async_copy.jl")
end

@testset "Events" begin
    include("events.jl")
end

@testset "Printing" begin
    include("print_test.jl")
end

@testset "Compiler" begin
    include("compiler.jl")
end

@testset "Reflection" begin
    include("reflection.jl")
end

include("examples.jl")
