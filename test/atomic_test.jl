using KernelAbstractions, Test

# Note: kernels affect second element because some CPU defaults will affect the
#       first element of a pointer if not specified, so I am covering the bases
@kernel function atomic_add_kernel!(input, b)
    atomic_add!(pointer(input,2),b) 
end

@kernel function atomic_sub_kernel!(input, b)
    atomic_sub!(pointer(input,2),b) 
end

@kernel function atomic_inc_kernel!(input, b)
    atomic_inc!(pointer(input,2),b) 
end

@kernel function atomic_dec_kernel!(input, b)
    atomic_dec!(pointer(input,2),b) 
end

@kernel function atomic_xchg_kernel!(input, b)
    atomic_xchg!(pointer(input,2),b) 
end

@kernel function atomic_and_kernel!(input, b)
    tid = @index(Global)
    atomic_and!(pointer(input),b[tid]) 
end

@kernel function atomic_or_kernel!(input, b)
    tid = @index(Global)
    atomic_or!(pointer(input),b[tid]) 
end

@kernel function atomic_xor_kernel!(input, b)
    tid = @index(Global)
    atomic_xor!(pointer(input),b[tid]) 
end

@kernel function atomic_max_kernel!(input, b)
    tid = @index(Global)
    atomic_max!(pointer(input,2), b[tid]) 
end

@kernel function atomic_min_kernel!(input, b)
    tid = @index(Global)
    atomic_min!(pointer(input,2), b[tid]) 
end

@kernel function atomic_cas_kernel!(input, b, c)
    atomic_cas!(pointer(input,2),b,c) 
end

function atomics_testsuite(backend, ArrayT)

    @testset "atomic addition tests" begin
        types = [Int32, Int64, UInt32, UInt64, Float32]

        for T in types
            A = ArrayT{T}([0,0])

            kernel! = atomic_add_kernel!(backend(), 4)
            wait(kernel!(A, one(T), ndrange=(1024)))

            @test Array(A)[2] == 1024
        end
    end

    @testset "atomic subtraction tests" begin
        types = [Int32, Int64, UInt32, UInt64, Float32]

        for T in types
            A = ArrayT{T}([2048,2048])

            kernel! = atomic_sub_kernel!(backend(), 4)
            wait(kernel!(A, one(T), ndrange=(1024)))

            @test Array(A)[2] == 1024
        end
    end

    @testset "atomic inc tests" begin
        types = [Int32]

        for T in types
            A = ArrayT{T}([0,0])

            kernel! = atomic_inc_kernel!(backend(), 4)
            wait(kernel!(A, T(512), ndrange=(768)))

            @test Array(A)[2] == 255
        end
    end

    @testset "atomic dec tests" begin
        types = [Int32]

        for T in types
            A = ArrayT{T}([1024,1024])

            kernel! = atomic_dec_kernel!(backend(), 4)
            wait(kernel!(A, T(512), ndrange=(256)))

            @test Array(A)[2] == 257
        end
    end

    @testset "atomic xchg tests" begin
        types = [Int32, Int64, UInt32, UInt64]

        for T in types
            A = ArrayT{T}([0,0])

            kernel! = atomic_xchg_kernel!(backend(), 4)
            wait(kernel!(A, T(1), ndrange=(256)))

            @test Array(A)[2] == one(T)
        end
    end

    @testset "atomic and tests" begin
        types = [Int32, Int64, UInt32, UInt64]

        for T in types
            A = ArrayT{T}([1023])
            B = ArrayT{T}([1023-2^(i-1) for i = 1:10])

            kernel! = atomic_and_kernel!(backend(), 4)
            wait(kernel!(A, B, ndrange=length(B)))
    
            @test Array(A)[1] == zero(T)
        end
    end

    @testset "atomic or tests" begin
        types = [Int32, Int64, UInt32, UInt64]

        for T in types
            A = ArrayT{T}([0])
            B = ArrayT{T}([2^(i-1) for i = 1:10])

            kernel! = atomic_or_kernel!(backend(), 4)
            wait(kernel!(A, B, ndrange=length(B)))
    
            @test Array(A)[1] == T(1023)
        end
    end

    @testset "atomic xor tests" begin
        types = [Int32, Int64, UInt32, UInt64]

        for T in types
            A = ArrayT{T}([1023])
            B = ArrayT{T}([2^(i-1) for i = 1:10])

            kernel! = atomic_xor_kernel!(backend(), 4)
            wait(kernel!(A, B, ndrange=length(B)))
    
            @test Array(A)[1] == T(0)
        end
    end

    @testset "atomic max tests" begin
        types = [Int32, Int64, UInt32, UInt64]

        for T in types
            A = ArrayT{T}([0,0])
            B = ArrayT{T}([i for i = 1:1024])

            kernel! = atomic_max_kernel!(backend(), 4)
            wait(kernel!(A, B, ndrange=length(B)))
    
            @test Array(A)[2] == T(1024)
        end
    end

    @testset "atomic min tests" begin
        types = [Int32, Int64, UInt32, UInt64]

        for T in types
            A = ArrayT{T}([1024,1024])
            B = ArrayT{T}([i for i = 1:1024])

            kernel! = atomic_min_kernel!(backend(), 4)
            wait(kernel!(A, B, ndrange=length(B)))
    
            @test Array(A)[2] == T(1)
        end
    end


    @testset "atomic cas tests" begin
        types = [Int32, Int64, UInt32, UInt64]

        for T in types
            A = ArrayT{T}([0,0])

            kernel! = atomic_cas_kernel!(backend(), 4)
            wait(kernel!(A, zero(T), one(T), ndrange=(1024)))

            @test Array(A)[2] == 1
        end
    end



end
