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

function test_typed_kernel_dynamic(backend, backend_str, ArrayT)
    A = ArrayT(ones(Float32, 1024, 1024))
    kernel = mul2(backend())
    res = if backend == CPU
        @ka_code_typed kernel(A, ndrange = size(A), workgroupsize = 16)
    else
        @ka_code_typed kernel(A, ndrange = size(A), workgroupsize = (32, 32))
    end
    if backend_str in ["CUDA", "ROCM", "oneAPI", "Metal", "OpenCL"]
        @test_broken isa(res, Pair{Core.CodeInfo, DataType})
    else
        @test isa(res, Pair{Core.CodeInfo, DataType})
    end
    @test isa(res[1].code, Array{Any, 1})
    return
end

function test_typed_kernel_dynamic_no_info(backend, backend_str, ArrayT)
    A = ArrayT(ones(Float32, 1024, 1024))
    B = similar(A)
    C = similar(A)
    kernel = add3(backend())
    res = @ka_code_typed kernel(A, B, C, ndrange = size(A))
    if backend_str in ["CUDA", "ROCM", "oneAPI", "Metal", "OpenCL"]
        @test_broken isa(res, Pair{Core.CodeInfo, DataType})
    else
        @test isa(res, Pair{Core.CodeInfo, DataType})
    end
    @test isa(res[1].code, Array{Any, 1})
    return
end

function test_typed_kernel_static(backend, backend_str, ArrayT)
    A = ArrayT(ones(Float32, 1024, 1024))
    kernel = if backend == CPU
        mul2(backend(), 16)
    else
        mul2(backend(), (32, 32))
    end
    res = @ka_code_typed kernel(A, ndrange = size(A))
    if backend_str in ["CUDA", "ROCM", "oneAPI", "Metal", "OpenCL"]
        @test_broken isa(res, Pair{Core.CodeInfo, DataType})
    else
        @test isa(res, Pair{Core.CodeInfo, DataType})
    end
    @test isa(res[1].code, Array{Any, 1})
    return
end

function test_typed_kernel_no_optimize(backend, backend_str, ArrayT)
    A = ArrayT(ones(Float32, 1024, 1024))
    kernel = if backend == CPU
        mul2(backend(), 16)
    else
        mul2(backend(), (32, 32))
    end
    res = @ka_code_typed optimize = false kernel(A, ndrange = size(A))
    res_opt = @ka_code_typed kernel(A, ndrange = size(A))
    # FIXME: Need a better test
    # @test size(res[1].code) < size(res_opt[1].code)
    return
end

function test_expr_kernel(backend, backend_str, ArrayT)
    A = ArrayT(ones(Float32, 1024, 1024))
    C = similar(A)
    kernel = if backend == CPU
        addi(backend())
    else
        addi(backend(), (32, 32))
    end
    res = @ka_code_typed kernel(A, C, 1 + 2, ndrange = size(A))
    if backend_str in ["CUDA", "ROCM", "oneAPI", "Metal", "OpenCL"]
        @test_broken isa(res, Pair{Core.CodeInfo, DataType})
    else
        @test isa(res, Pair{Core.CodeInfo, DataType})
    end
    @test isa(res[1].code, Array{Any, 1})
    return
end

function reflection_testsuite(backend, backend_str, ArrayT)
    test_typed_kernel_dynamic(backend, backend_str, ArrayT)
    test_typed_kernel_dynamic_no_info(backend, backend_str, ArrayT)
    test_typed_kernel_static(backend, backend_str, ArrayT)
    test_typed_kernel_no_optimize(backend, backend_str, ArrayT)
    test_expr_kernel(backend, backend_str, ArrayT)
    return
end
