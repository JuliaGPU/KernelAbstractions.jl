using KernelAbstractions
using Test

import KernelAbstractions.NDIteration: NDRange, StaticSize, DynamicSize

@kernel function index(A)
    I  = @index(Global, NTuple)
    @show A[I...]
end
kernel = index(CPU(), DynamicSize(), DynamicSize())
iterspace = NDRange{1, StaticSize{(128,)}, StaticSize{(8,)}}();
ctx = KernelAbstractions.mkcontext(kernel, 1, nothing, iterspace, Val(KernelAbstractions.NoDynamicCheck()))

@test KernelAbstractions.Cassette.overdub(ctx, KernelAbstractions.__index_Global_NTuple, CartesianIndex(1)) == (1,)

@kernel function literal_pow(A)
    A[1] = 2^11
end

@kernel function checked(A, a, b)
    A[1] = Base.Checked.checked_add(a, b)
end

let (CI, rt) = @ka_code_typed literal_pow(CPU())(zeros(Int,1), ndrange=1)
    # test that there is no invoke of overdub
    @test !any(stmt->(stmt isa Expr) && stmt.head == :invoke, CI.code)
end

if VERSION >= v"1.5"
    let (CI, rt) = @ka_code_typed checked(CPU())(zeros(Int,1), 1, 2, ndrange=1)
        # test that there is no invoke of overdub
        @test !any(stmt->(stmt isa Expr) && stmt.head == :invoke, CI.code)
    end
end