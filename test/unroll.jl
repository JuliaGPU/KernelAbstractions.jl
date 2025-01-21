using KernelAbstractions
using KernelAbstractions.Extras
using StaticArrays

@kernel function kernel_unroll!(a)
    @unroll for i in 1:5
        @inbounds a[i] = i
    end
end

@kernel function kernel_unroll!(a, ::Val{N}) where {N}
    let M = N + 5
        @unroll for i in 6:M
            @inbounds a[i - 5] = i
        end
        @synchronize
    end
end

# Check that nested `@unroll` doesn't throw a syntax error
@kernel function kernel_unroll2!(A)
    @uniform begin
        a = MVector{3, Float32}(1, 2, 3)
        b = MVector{3, Float32}(3, 2, 1)
        c = MMatrix{3, 3, Float32}(undef)
    end
    I = @index(Global)
    @inbounds for m in 1:3
        @unroll for j in 1:3
            @unroll for i in 1:3
                c[1, j] = m * a[1] * b[j]
            end
        end
        A[I] = c[1, 1]
        @synchronize(m % 2 == 0)
    end
end

function unroll_testsuite(backend, ArrayT)
    a = ArrayT(zeros(Float32, 5))
    kernel! = kernel_unroll!(backend(), 1, 1)
    kernel!(a)
    kernel!(a, Val(5))
    kernel2! = kernel_unroll2!(backend(), 1, 1)
    kernel2!(a)
    synchronize(backend())
    return
end
