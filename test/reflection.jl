using KernelAbstractions, Test

@kernel function mul2(A)
    I = @index(Global, Cartesian)
    A[I] = 2 * A[I]
end

@kernel function add3(A, B, C)
    I = @index(Global, Cartesian)
    A[I] = B[I] + C[I]
end

@kernel function addi(A, C, i)
    I = @index(Global, Cartesian)
    A[I] = i + C[I]
end

function test_typed_kernel_dynamic(backend, ArrayT)
    A = ArrayT(ones(1024, 1024))
    kernel = mul2(backend())
    res = if backend == CPU
        @ka_code_typed kernel(A, ndrange=size(A), workgroupsize=16)
    else
        @ka_code_typed kernel(A, ndrange=size(A), workgroupsize=(32, 32))
    end
    @test isa(res, Pair{Core.CodeInfo, DataType})
    @test isa(res[1].code, Array{Any,1})
end

function test_typed_kernel_dynamic_no_info(backend, ArrayT)
    A = ArrayT(ones(1024, 1024))
    B = similar(A)
    C = similar(A)
    kernel = add3(backend())
    res = @ka_code_typed kernel(A, B, C, ndrange=size(A))
    @test isa(res, Pair{Core.CodeInfo, DataType})
    @test isa(res[1].code, Array{Any,1})
end

function test_typed_kernel_static(backend, ArrayT)
    A = ArrayT(ones(1024, 1024))
    kernel = if backend == CPU
        mul2(backend(), 16)
    else
        mul2(backend(), (32, 32))
    end
    res = @ka_code_typed kernel(A, ndrange=size(A))
    @test isa(res, Pair{Core.CodeInfo, DataType})
    @test isa(res[1].code, Array{Any,1})
end

function test_typed_kernel_no_optimize(backend, ArrayT)
    A = ArrayT(ones(1024, 1024))
    B = similar(A)
    C = similar(A)
    kernel = if backend == CPU
        add3(backend(), 16)
    else
        add3(backend(), (32, 32))
    end
    res = @ka_code_typed optimize=false kernel(A, ndrange=size(A))
    @test isa(res, Pair{Core.CodeInfo,Core.TypeofBottom})
    res_opt = @ka_code_typed kernel(A, ndrange=size(A))
    @test size(res[1].code) < size(res_opt[1].code)
end

function test_expr_kernel(backend, ArrayT)
    A = ArrayT(ones(1024, 1024))
    C = similar(A)
    kernel = if backend == CPU
        addi(backend())
    else
        addi(backend(), (32, 32))
    end
    res = @ka_code_typed kernel(A, C, 1+2, ndrange=size(A))
    @test isa(res, Pair{Core.CodeInfo, DataType})
    @test isa(res[1].code, Array{Any,1})
end

function reflection_testsuite(backend, ArrayT)
    test_typed_kernel_dynamic(backend, ArrayT)
    test_typed_kernel_dynamic_no_info(backend, ArrayT)
    test_typed_kernel_static(backend, ArrayT)
    test_typed_kernel_no_optimize(backend, ArrayT)
    test_expr_kernel(backend, ArrayT)
end
