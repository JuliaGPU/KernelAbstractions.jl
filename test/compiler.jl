using KernelAbstractions
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

@kernel function pow(A, B)
    A[1] = A[1]^B[1]
end

@kernel function checked(A, a, b)
    A[1] = Base.Checked.checked_add(a, b)
end

function check_for_overdub(stmt)
    if stmt isa Expr
        if stmt.head == :invoke
            mi = first(stmt.args)::Core.MethodInstance
            if mi.def.name === :overdub
                @show stmt
                return true
            end
        end
    end
    return false
end

function compiler_testsuite(backend, ArrayT)
    kernel = index(CPU(), DynamicSize(), DynamicSize())
    iterspace = NDRange{1, StaticSize{(128,)}, StaticSize{(8,)}}()
    ctx = KernelAbstractions.mkcontext(kernel, 1, nothing, iterspace, Val(KernelAbstractions.NoDynamicCheck()))
    @test KernelAbstractions.__index_Global_NTuple(ctx, CartesianIndex(1)) == (1,)

    A = ArrayT{Int}(undef, 1)
    let (CI, rt) = @ka_code_typed literal_pow(backend())(A, ndrange = 1)
        # test that there is no invoke of overdub
        @test !any(check_for_overdub, CI.code)
    end

    A = ArrayT{Float32}(undef, 1)
    let (CI, rt) = @ka_code_typed square(backend())(A, A, ndrange = 1)
        # test that there is no invoke of overdub
        @test !any(check_for_overdub, CI.code)
    end

    A = ArrayT{Float32}(undef, 1)
    B = ArrayT{Float32}(undef, 1)
    let (CI, rt) = @ka_code_typed pow(backend())(A, B, ndrange = 1)
        # test that there is no invoke of overdub
        @test !any(check_for_overdub, CI.code)
    end

    A = ArrayT{Float32}(undef, 1)
    B = ArrayT{Int32}(undef, 1)
    let (CI, rt) = @ka_code_typed pow(backend())(A, B, ndrange = 1)
        # test that there is no invoke of overdub
        @test !any(check_for_overdub, CI.code)
    end

    A = ArrayT{Int}(undef, 1)
    let (CI, rt) = @ka_code_typed checked(backend())(A, 1, 2, ndrange = 1)
        # test that there is no invoke of overdub
        @test !any(check_for_overdub, CI.code)
    end
    return
end
