using KernelAbstractions, Test
using CUDA

if has_cuda_gpu()
    CUDA.allowscalar(false)
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
    if has_cuda_gpu()
        wait(test_print(CUDADevice()))
        @test true
    end

    wait(test_print(CPU()))
    @test true

    @print("Why this should work\n")
    @test true
end
