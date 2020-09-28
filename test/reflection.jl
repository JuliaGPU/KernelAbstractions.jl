using KernelAbstractions, Test, CUDA

@kernel function mul2(A)
    I = @index(Global)
    A[I] = 2 * A[I]
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
    kernel = mul2(CPU())
    res = @ka_code_typed kernel(A, ndrange=size(A))
    @test isa(res, Pair{Core.CodeInfo, DataType})
    @test isa(res[1].code, Array{Any,1})

    if has_cuda_gpu()
        A = CUDA.ones(1024, 1024)
        kernel = mul2(CUDADevice())
        res = @ka_code_typed kernel(A, ndrange=size(A))
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
    kernel = mul2(CPU(), 16)
    res = @ka_code_typed optimize=false kernel(A, ndrange=size(A))
    @test isa(res, Pair{Core.CodeInfo, DataType})
    res_opt = @ka_code_typed kernel(A, ndrange=size(A))
    @test size(res[1].code) < size(res_opt[1].code)

    if has_cuda_gpu()
        A = CUDA.ones(1024, 1024)
        kernel = mul2(CUDADevice(), (32, 32))
        res = @ka_code_typed optimize=false kernel(A, ndrange=size(A))
        @test isa(res, Pair{Core.CodeInfo, DataType})
        res_opt = @ka_code_typed kernel(A, ndrange=size(A))
        @test size(res[1].code) < size(res_opt[1].code)
    end
end

test_typed_kernel_dynamic()
test_typed_kernel_dynamic_no_info()
test_typed_kernel_static()
test_typed_kernel_no_optimize()
