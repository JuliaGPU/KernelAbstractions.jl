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
  if backend == CPU()
    kernel = kernel_print(CPU(), 4)
  else
    kernel = kernel_print(CUDA(), 4)
  end
  kernel(ndrange=(4,)) 
end


@testset "print test" begin
    if CUDAapi.has_cuda_gpu()
        wait(test_print(CUDA()))
        @test true
    end

    wait(test_print(CPU()))
    @test true
end
