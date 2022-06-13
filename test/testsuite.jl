module Testsuite

using ..KernelAbstractions
using ..Test

include("test.jl")
include("localmem.jl")
include("private.jl")
include("unroll.jl")
include("nditeration.jl")
include("async_copy.jl")
include("events.jl")
include("print_test.jl")
include("compiler.jl")
include("reflection.jl")
include("examples.jl")
include("convert.jl")
include("atomic_test.jl")
include("atomix_test.jl")

function testsuite(backend, backend_str, backend_mod, AT, DAT)
    @testset "Unittests" begin
        unittest_testsuite(backend, backend_str, backend_mod, AT, DAT)
    end

    @testset "Localmem" begin
        localmem_testsuite(backend, AT)
    end

    @testset "Private" begin
        private_testsuite(backend, AT)
    end

    if backend_str != "ROCM"
        @testset "Unroll" begin
            unroll_testsuite(backend, AT)
        end
    end

    @testset "NDIteration" begin
        nditeration_testsuite()
    end

    if backend_str != "ROCM"
        @testset "async_copy!" begin
            asynccopy_testsuite(backend, AT)
        end
    end

    @testset "Events" begin
        events_testsuite()
    end

    if backend_str != "ROCM"
        @testset "Printing" begin
            printing_testsuite(backend)
        end
    end

    if backend_str != "ROCM"
        @testset "Compiler" begin
            compiler_testsuite(backend, AT)
        end
    end

    @testset "Reflection" begin
        reflection_testsuite(backend, backend_str, AT)
    end

    @testset "Convert" begin
        convert_testsuite(backend, AT)
    end

    @testset "Atomix" begin
        atomix_testsuite(backend, AT)
    end

    if backend_str == "CUDA" || backend_str == "ROCM"
        @testset "Examples" begin
            examples_testsuite(backend_str)
        end
    end
end

end
