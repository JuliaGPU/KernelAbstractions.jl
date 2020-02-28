using KernelAbstractions
using KernelAbstractions.Extras
using StaticArrays

@kernel function kernel_unroll!(a)
  @unroll for i in 1:5
    @inbounds a[i] = i
  end
end

@kernel function kernel_unroll!(a, ::Val{N}) where N
  @unroll for i in 1:N
    @inbounds a[i] = i
  end
end

# Check that nested `@unroll` doesn't throw a syntax error
@kernel function kernel_unroll!(a, ::Val{N}) where N
  @uniform begin
    a = MVector{3, Float64}(1, 2, 3)
    b = MVector{3, Float64}(3, 2, 1)
    c = MMatrix{3, 3, Float64}(undef)
  end
  I = @index(Global)
  @inbounds for m in 1:3
    @unroll for j = 1:3
      @unroll for i = 1:3
        c[1, j] = m * a[1] * b[j]
      end
    end
    a[I] = c[1, 1]
    m % 2 == 0 && @synchronize
  end
end

let
  a = zeros(5)
  kernel! = kernel_unroll!(CPU(), 1, 1)
  event = kernel!(a)
  wait(event)
  event = kernel!(a, Val(5))
  wait(event)
end
