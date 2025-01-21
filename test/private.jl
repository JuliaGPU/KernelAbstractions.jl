using KernelAbstractions
using Test

@kernel function stmt_form()
    @uniform bs = @groupsize()[1]
    @private s = bs ÷ 2
    @synchronize
end

@kernel function typetest(A, B)
    priv = @private eltype(A) (1,)
    I = @index(Global, Linear)
    @inbounds begin
        B[I] = eltype(priv) === eltype(A)
    end
end

@kernel function private(A)
    @uniform N = prod(@groupsize())
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
@kernel function forloop(A, ::Val{N}) where {N}
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

function private_testsuite(backend, ArrayT)
    @testset "kernels" begin
        stmt_form(backend(), 16)(ndrange = 16)
        synchronize(backend())
        A = ArrayT{Int}(undef, 64)
        private(backend(), 16)(A, ndrange = size(A))
        synchronize(backend())
        B = Array(A)
        @test all(B[1:16] .== 16:-1:1)
        @test all(B[17:32] .== 16:-1:1)
        @test all(B[33:48] .== 16:-1:1)
        @test all(B[49:64] .== 16:-1:1)

        A = ArrayT{Int}(undef, 64, 64)
        A .= 1
        forloop(backend())(A, Val(size(A, 2)), ndrange = size(A, 1), workgroupsize = size(A, 1))
        synchronize(backend())
        @test all(Array(A)[:, 1] .== 64)
        @test all(Array(A)[:, 2:end] .== 1)

        B = ArrayT{Bool}(undef, size(A)...)
        typetest(backend(), 16)(A, B, ndrange = size(A))
        synchronize(backend())
        @test all(Array(B))

        A = ArrayT(ones(Float32, 64, 3))
        out = ArrayT{Float32}(undef, 64)
        reduce_private(backend(), 8)(out, A, ndrange = size(out))
        synchronize(backend())
        @test all(Array(out) .== 3.0f0)
    end

    if backend == CPU
        @testset "codegen" begin
            IR = sprint() do io
                KernelAbstractions.ka_code_llvm(
                    io, reduce_private(backend(), (8,)), Tuple{ArrayT{Float64, 1}, ArrayT{Float64, 2}},
                    optimize = true, ndrange = (64,),
                )
            end
            if VERSION >= v"1.11-" && Base.JLOptions().check_bounds > 0
                # JuliaGPU/KernelAbstractions.jl#528
                @test_broken !occursin("gcframe", IR)
            else
                @test !occursin("gcframe", IR)
            end
        end
    end
    return
end
