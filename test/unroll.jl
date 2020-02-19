using KernelAbstractions
using KernelAbstractions.Extras

@kernel function kernel_unroll!(a)
  @unroll for i in 1:5
    @inbounds a[i] = i
  end
end

let
  a = zeros(5)
  kernel! = kernel_unroll!(CPU(), 1, 1)
  event = kernel!(a)
  wait(event)
end
