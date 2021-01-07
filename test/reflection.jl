using KernelAbstractions, Test, CUDA

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

function test_typed_kernel_dynamic()
    A = ones(1024, 1024)
    kernel = mul2(CPU())
    res = @ka_code_typed kernel(A, ndrange=size(A), workgroupsize=16)
    @test isa(res, Pair{Core.CodeInfo, DataType})
    @test isa(res[1].code, Array{Any,1})

    if has_cuda_gpu()
        A = CUDA.ones(1024, 1024)
        kernel = mul2(CUDADevice())
        res = @ka_code_typed kernel(A, ndrange=size(A), workgroupsize=(32, 32))
        @test isa(res, Pair{Core.CodeInfo, DataType})
        @test isa(res[1].code, Array{Any,1})
    end
end

function test_typed_kernel_dynamic_no_info()
    A = ones(1024, 1024)
    B = similar(A)
    C = similar(A)
    kernel = add3(CPU())
    res = @ka_code_typed kernel(A, B, C, ndrange=size(A))
    @test isa(res, Pair{Core.CodeInfo, DataType})
    @test isa(res[1].code, Array{Any,1})

    if has_cuda_gpu()
        A = CUDA.ones(1024, 1024)
        B = similar(A)
        C = similar(A)
        kernel = add3(CUDADevice())
        res = @ka_code_typed kernel(A, B, C, ndrange=size(A))
        @test isa(res, Pair{Core.CodeInfo, DataType})
        @test isa(res[1].code, Array{Any,1})
    end
end

function test_typed_kernel_static()
    A = ones(1024, 1024)
    kernel = mul2(CPU(), 16)
    res = @ka_code_typed kernel(A, ndrange=size(A))
    @test isa(res, Pair{Core.CodeInfo, DataType})
    @test isa(res[1].code, Array{Any,1})

    if has_cuda_gpu()
        A = CUDA.ones(1024, 1024)
        kernel = mul2(CUDADevice(), (32, 32))
        res = @ka_code_typed kernel(A, ndrange=size(A))
        @test isa(res, Pair{Core.CodeInfo, DataType})
        @test isa(res[1].code, Array{Any,1})
    end
end

function test_typed_kernel_no_optimize()
    A = ones(1024, 1024)
    B = similar(A)
    C = similar(A)
    kernel = add3(CPU(), 16)
    res = @ka_code_typed optimize=false kernel(A, ndrange=size(A))
    @test isa(res, Pair{Core.CodeInfo,Core.TypeofBottom})
    res_opt = @ka_code_typed kernel(A, ndrange=size(A))
    @test size(res[1].code) < size(res_opt[1].code)

    if has_cuda_gpu()
        A = CUDA.ones(1024, 1024)
        B = similar(A)
        C = similar(A)
        kernel = add3(CUDADevice(), (32, 32))
        res = @ka_code_typed optimize=false kernel(A, ndrange=size(A))
        @test isa(res, Pair{Core.CodeInfo,Core.TypeofBottom})
        res_opt = @ka_code_typed kernel(A, ndrange=size(A))
        @test size(res[1].code) < size(res_opt[1].code)
    end
end

function test_expr_kernel()
    A = ones(1024, 1024)
    C = similar(A)
    kernel = addi(CPU())
    res = @ka_code_typed kernel(A, C, 1+2, ndrange=size(A))
    @test isa(res, Pair{Core.CodeInfo, DataType})
    @test isa(res[1].code, Array{Any,1})

    if has_cuda_gpu()
        A = CUDA.ones(1024, 1024)
        C = similar(A)
        kernel = addi(CUDADevice(), (32, 32))
        res = @ka_code_typed kernel(A, C, 1+2, ndrange=size(A))
        @test isa(res, Pair{Core.CodeInfo, DataType})
        @test isa(res[1].code, Array{Any,1})
    end
end

test_typed_kernel_dynamic()
test_typed_kernel_dynamic_no_info()
test_typed_kernel_static()
test_typed_kernel_no_optimize()
test_expr_kernel()
