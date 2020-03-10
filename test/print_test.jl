using KernelAbstractions, Test
import KernelAbstractions: @print
@kernel function kernel_print()
  I = @index(Global)
  @print("Hello from thread ", I)
end

function test_print(gpu_avail)
  if gpu_avail
    kernel = kernel_print(CUDA(), 256)
  else
    kernel = kernel_print(CPU(), 4)
  end
  kernel(ndrange=(256,)) 
end


@testset "print test" begin
#=
  @static if Base.find_package("CuArrays") !== nothing
    using CuArrays
    using CUDAnative

    test_print(true)
  end
=#
    wait(test_print(false))
end
