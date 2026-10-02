# Run by runtests.jl in a fresh process with several threads: it checks how the CPU back-end
# initializes its state, and how tasks on different threads share it.

using KernelAbstractions
using Test

import KernelAbstractions.POCL: POCL, @opencl

@kernel function fill_value!(A, v)
    I = @index(Global, Linear)
    @inbounds A[I] = v
end

@kernel function fill_rand!(A)
    I = @index(Global, Linear)
    @inbounds A[I] = rand(Float32)
end

empty_kernel() = return

function pointer_value(out, p)
    @inbounds out[1] = UInt(reinterpret(Ptr{Cvoid}, p))
    return
end

@testset "POCL session" begin
    @test Threads.nthreads() > 1

    @testset "shared context, per-task queues" begin
        # concurrent first use initializes the back-end once
        sessions = fetch.([Threads.@spawn POCL.session() for _ in 1:16])
        @test all(s -> s === POCL.session(), sessions)

        queues = fetch.([Threads.@spawn (POCL.queue(), POCL.queue()) for _ in 1:16])
        @test all(((a, b),) -> a === b, queues)
        @test allunique(first.(queues))
    end

    @testset "kernels are linked once" begin
        kernels = fetch.([Threads.@spawn POCL.clfunction(empty_kernel).fun for _ in 1:16])
        @test all(k -> k === POCL.clfunction(empty_kernel).fun, kernels)
    end

    # kernel arguments are state of the kernel object, which all tasks share
    @testset "concurrent launches of one kernel" begin
        N = 4096
        arrays = [zeros(Float32, N) for _ in 1:32]
        @sync for (j, A) in enumerate(arrays)
            Threads.@spawn for _ in 1:10
                fill_value!(CPU())(A, Float32(j); ndrange = N)
            end
        end
        @test all(((j, A),) -> all(==(j), A), enumerate(arrays))

        # kernels using the RNG get extra arguments, sized by the workgroup size
        arrays = [fill(-1.0f0, N) for _ in 1:32]
        @sync for (j, A) in enumerate(arrays)
            Threads.@spawn for _ in 1:10
                fill_rand!(CPU(), isodd(j) ? 16 : 64)(A; ndrange = N)
            end
        end
        @test all(A -> all(x -> 0 <= x < 1, A), arrays)
    end

    @testset "null pointer arguments" begin
        out = UInt[1]
        x = Float32[0]
        P = Core.LLVMPtr{Float32, 1}
        function launch(p)
            GC.@preserve x @opencl pointer_value(out, p)
            return out[1]
        end
        @test launch(reinterpret(P, pointer(x))) == UInt(pointer(x))
        # the kernel mustn't see the pointer from the previous launch
        @test launch(reinterpret(P, C_NULL)) == 0
        # before Julia 1.12, a `Ptr` argument is an integer
        out[1] = 1
        @test launch(Ptr{Float32}(C_NULL)) == 0
    end

    @testset "reset" begin
        s = POCL.session()
        q = POCL.queue()
        POCL.reset_session_state!()
        @test POCL.session() !== s
        @test POCL.queue() !== q
        A = zeros(Float32, 16)
        fill_value!(CPU())(A, 1.0f0; ndrange = 16)
        @test all(==(1), A)
    end
end
