@kernel function kernel_print()
  I = @index(Global)
  @println("Hello from thread", I)
end

function test_print!(gpu_avail)
  if isa(false)
    kernel! = matmul_kernel!(CPU())
  else
    kernel! = matmul_kernel!(CUDA())
  end
  kernel!(ndrange=size(3)) 
end


@testset "print test" begin
  @static if Base.find_package("CuArrays") !== nothing
    using CuArrays
    using CUDAnative

    test_print(true)
  end
    test_print(false)
end
