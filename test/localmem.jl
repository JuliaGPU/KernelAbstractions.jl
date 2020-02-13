using KernelAbstractions
using Test
using CUDAapi
if has_cuda_gpu()
    using CuArrays
    CuArrays.allowscalar(false)
end

@kernel function dynamic(A)
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    lmem = @dynamic_localmem eltype(A) (wkrgpsize)->2*wkrgpsize
    lmem[2*i] = A[I]
    @synchronize
    A[I] = lmem[2*i]
end

@kernel function dynamic2(A)
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    lmem  = @dynamic_localmem eltype(A) (wkrgpsize)->2*wkrgpsize
    lmem2 = @dynamic_localmem Int (wkrgpsize)->wkrgpsize
    lmem2[i] = i
    lmem[2*i] = A[I]
    @synchronize
    A[I] = lmem[2*lmem2[i]]
end

@kernel function dynamic_mixed(A)
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    lmem  = @dynamic_localmem eltype(A) (wkrgpsize)->2*wkrgpsize
    lmem2 = @localmem Int groupsize() # Ok iff groupsize is static 
    lmem2[i] = i
    lmem[2*i] = A[I]
    @synchronize
    A[I] = lmem[2*lmem2[i]]
end


function harness(backend, ArrayT)
    A = ArrayT{Float64}(undef, 16, 16)
    A .= 1.0
    wait(dynamic(backend, 16)(A, ndrange=size(A)))

    A = ArrayT{Float64}(undef, 16, 16)
    wait(dynamic2(backend, 16)(A, ndrange=size(A)))

    A = ArrayT{Float64}(undef, 16, 16)
    wait(dynamic_mixed(backend, 16)(A, ndrange=size(A)))
end

@testset "kernels" begin
    harness(CPU(), Array)
    if has_cuda_gpu()
        harness(CUDA(), CuArray)
    end
end
