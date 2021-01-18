using KernelAbstractions
using Test
using CUDA

if has_cuda_gpu()
    CUDA.allowscalar(false)
end

@kernel function typetest(A, B)
    priv = @private eltype(A) (1,)
    I = @index(Global, Linear)
    @inbounds begin
      B[I] = eltype(priv) === eltype(A)
    end
end

@kernel function private(A)
    @uniform N = prod(groupsize())
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    priv = @private Int (1,)
    @inbounds begin
        priv[1] = N - i + 1
        @synchronize
        A[I] = priv[1]
    end
end

# This is horrible don't write code like this
@kernel function forloop(A, ::Val{N}) where N
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    priv = @private Int (N,)
    for j in 1:N
        priv[j] = A[I, j]
    end
    A[I, 1] = 0
    @synchronize
    for j in 1:N
        k = mod1(j + i - 1, N)
        A[k, 1] += priv[j]
        @synchronize
    end
end

@kernel function reduce_private(out, A)
    I = @index(Global, NTuple)
    i = @index(Local)

    priv = @private eltype(A) (1,)
    @inbounds begin
      priv[1] = zero(eltype(A))
      for k in 1:size(A, ndims(A))
        priv[1] += A[I..., k]
      end
      out[I...] = priv[1]
    end
end

function harness(backend, ArrayT)
    A = ArrayT{Int}(undef, 64)
    wait(private(backend, 16)(A, ndrange=size(A)))
    @test all(A[1:16] .== 16:-1:1)
    @test all(A[17:32] .== 16:-1:1)
    @test all(A[33:48] .== 16:-1:1)
    @test all(A[49:64] .== 16:-1:1)

    A = ArrayT{Int}(undef, 64, 64)
    A .= 1
    wait(forloop(backend)(A, Val(size(A, 2)), ndrange=size(A,1), workgroupsize=size(A,1)))
    @test all(A[:, 1] .== 64)
    @test all(A[:, 2:end] .== 1)

    B = ArrayT{Bool}(undef, size(A)...)
    wait(typetest(backend, 16)(A, B, ndrange=size(A)))
    @test all(B)

    A = ArrayT{Float64}(ones(64,3));
    out =  ArrayT{Float64}(undef, 64)
    wait(reduce_private(CPU(), 8)(out, A, ndrange=size(out)))
    @test all(out .== 3.0)
end

@testset "kernels" begin
    harness(CPU(), Array)
    if has_cuda_gpu()
        harness(CUDADevice(), CuArray)
    end
end

@testset "codegen" begin
    IR = sprint() do io
        KernelAbstractions.ka_code_llvm(io, reduce_private(CPU(), (8,)), Tuple{Vector{Float64}, Matrix{Float64}},
                                        optimize=true, ndrange=(64,))
    end
    @test !occursin("gcframe", IR)
end