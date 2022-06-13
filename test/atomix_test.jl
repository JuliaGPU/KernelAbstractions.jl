using KernelAbstractions, Test
using KernelAbstractions: @atomic, @atomicswap, @atomicreplace

# Note: kernels affect second element because some CPU defaults will affect the
#       first element of a pointer if not specified, so I am covering the bases
@kernel function atomix_add_kernel!(input, b)
    @atomic input[2] += b 
end

@kernel function atomix_sub_kernel!(input, b)
    @atomic input[2] -= b 
end

@kernel function atomix_xchg_kernel!(input, b)
    @atomicswap input[2] = b
end

@kernel function atomix_max_kernel!(input, b)
    tid = @index(Global)
    @atomic max(input[2], b[tid]) 
end

@kernel function atomix_min_kernel!(input, b)
    tid = @index(Global)
    @atomic min(input[2], b[tid]) 
end

@kernel function atomix_cas_kernel!(input, b, c)
    @atomicreplace input[2] b => c
    #atomic_cas!(pointer(input,2),b,c) 
end

function atomix_testsuite(backend, ArrayT)

    @testset "atomic addition tests" begin
        types = [Int32, Int64, UInt32, UInt64, Float32]

        for T in types
            A = ArrayT{T}([0,0])

            kernel! = atomix_add_kernel!(backend(), 4)
            wait(kernel!(A, one(T), ndrange=(1024)))

            @test Array(A)[2] == 1024
        end
    end

    @testset "atomic subtraction tests" begin
        types = [Int32, Int64, UInt32, UInt64, Float32]

        for T in types
            A = ArrayT{T}([2048,2048])

            kernel! = atomix_sub_kernel!(backend(), 4)
            wait(kernel!(A, one(T), ndrange=(1024)))

            @test Array(A)[2] == 1024
        end
    end

    @testset "atomic xchg tests" begin
        types = [Int32, Int64, UInt32, UInt64]

        for T in types
            A = ArrayT{T}([0,0])

            kernel! = atomix_xchg_kernel!(backend(), 4)
            wait(kernel!(A, T(1), ndrange=(256)))

            @test Array(A)[2] == one(T)
        end
    end

    @testset "atomic max tests" begin
        types = [Int32, Int64, UInt32, UInt64]

        for T in types
            A = ArrayT{T}([0,0])
            B = ArrayT{T}([i for i = 1:1024])

            kernel! = atomix_max_kernel!(backend(), 4)
            wait(kernel!(A, B, ndrange=length(B)))
    
            @test Array(A)[2] == T(1024)
        end
    end

    @testset "atomic min tests" begin
        types = [Int32, Int64, UInt32, UInt64]

        for T in types
            A = ArrayT{T}([1024,1024])
            B = ArrayT{T}([i for i = 1:1024])

            kernel! = atomix_min_kernel!(backend(), 4)
            wait(kernel!(A, B, ndrange=length(B)))
    
            @test Array(A)[2] == T(1)
        end
    end


    @testset "atomic cas tests" begin
        types = [Int32, Int64, UInt32, UInt64]

        for T in types
            A = ArrayT{T}([0,0])

            kernel! = atomix_cas_kernel!(backend(), 4)
            wait(kernel!(A, zero(T), one(T), ndrange=(1024)))

            @test Array(A)[2] == 1
        end
    end
end
