using KernelAbstractions, Test
using CUDA

if has_cuda_gpu()
    CUDA.allowscalar(false)
end

struct Foo{A,B} end

@kernel function kernel_print()
  I = @index(Global)
  @print("Hello from thread ", I, "!\n")
end

@kernel function kernel_printf()
  I = @index(Global)
  @printf("Hello printf %s thread %d! type = %s.\n", "from", I, nameof(Foo))
end

function test_print(backend)
  kernel = kernel_print(backend, 4)
  kernel(ndrange=(4,))
end

function test_printf(backend)
  kernel = kernel_printf(backend, 4)
  kernel(ndrange=(4,))
end

@testset "print test" begin
    wait(test_print(CPU()))
    @test true

    wait(test_printf(CPU()))
    @test true

    if has_cuda_gpu()
        wait(test_print(CUDADevice()))
        @test true
        wait(test_printf(CUDADevice()))
        @test true
    end

    @print("Why this should work")
    @test true

    @printf("Why this should work")
    @test true
end

