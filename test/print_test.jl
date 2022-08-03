using KernelAbstractions, Test

@kernel function kernel_print()
  I = @index(Global)
  @print("Hello from thread ", I, "!\n")
end

function printing_testsuite(backend)
    @testset "print test" begin
        kernel = kernel_print(backend(), 4)
        kernel(ndrange=(4,))
        synchronize(backend())
        @test true

        @print("Why this should work\n")
        @test true
    end
end
