using KernelAbstractions
using CUDAKernels
using Test

import KernelAbstractions.NDIteration: NDRange, StaticSize, DynamicSize

@kernel function index(A)
    I = @index(Global, NTuple)
    @show A[I...]
end

@kernel function literal_pow(A)
    A[1] = 2^11
end

@kernel function square(A, B)
    A[1] = B[1]^2
end

@kernel function checked(A, a, b)
    A[1] = Base.Checked.checked_add(a, b)
end

function compiler_testsuite()
    kernel = index(CPU(), DynamicSize(), DynamicSize())
    iterspace = NDRange{1, StaticSize{(128,)}, StaticSize{(8,)}}();
    ctx = KernelAbstractions.mkcontext(kernel, 1, nothing, iterspace, Val(KernelAbstractions.NoDynamicCheck()))
    CTX = KernelAbstractions.cassette(kernel)
    @test KernelAbstractions.Cassette.overdub(CTX, KernelAbstractions.__index_Global_NTuple, ctx, CartesianIndex(1)) == (1,)

    let (CI, rt) = @ka_code_typed literal_pow(CPU())(zeros(Int,1), ndrange=1)
        # test that there is no invoke of overdub
        @test !any(stmt->(stmt isa Expr) && stmt.head == :invoke, CI.code)
    end

    let (CI, rt) = @ka_code_typed square(CUDADevice())(zeros(1), zeros(1), ndrange=1)
        # test that there is no invoke of overdub
        @test !any(stmt->(stmt isa Expr) && stmt.head == :invoke, CI.code)
    end

    if VERSION >= v"1.5"
        let (CI, rt) = @ka_code_typed checked(CPU())(zeros(Int,1), 1, 2, ndrange=1)
            # test that there is no invoke of overdub
            @test !any(stmt->(stmt isa Expr) && stmt.head == :invoke, CI.code)
        end
    end
end
