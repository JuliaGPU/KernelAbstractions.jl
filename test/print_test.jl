using KernelAbstractions, Test
using CUDAapi
if CUDAapi.has_cuda_gpu()
    using CuArrays
    CuArrays.allowscalar(false)
end

@kernel function kernel_print()
  I = @index(Global)
  @print("Hello from thread ", I, "!\n")
end

function test_print(backend)
  kernel = kernel_print(backend, 4)
  kernel(ndrange=(4,)) 
end

@testset "print test" begin
    if CUDAapi.has_cuda_gpu()
        wait(test_print(CUDA()))
        @test true
    end

    wait(test_print(CPU()))
    @test true

    @print("Why this should work")
    @test true
end
