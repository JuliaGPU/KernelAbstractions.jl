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

function test_print(gpu_avail)
  if gpu_avail
    kernel = kernel_print(CUDA(), 4)
  else
    kernel = kernel_print(CPU(), 4)
  end
  kernel(ndrange=(4,)) 
end


@testset "print test" begin
    if CUDAapi.has_cuda_gpu()
        wait(test_print(true))
        @test true
    end

    wait(test_print(false))
    @test true
end
