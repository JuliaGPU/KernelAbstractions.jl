using KernelAbstractions, Test

@kernel function kernel_print()
    I = @index(Global)
    @print("Hello from thread ", I, "!\n")
end

function printing_testsuite(backend)
    # TODO: capture output and verify it (this requires the ability to check for
    #       back-end capabilities, as not all back-ends support printing)

    @testset "print test" begin
        kernel = kernel_print(backend(), 4)
        redirect_stdout(devnull) do
            kernel(ndrange = (4,))
            synchronize(backend())
        end
        @test true

        redirect_stdout(devnull) do
            @print("Why this should work\n")
            synchronize(backend())
        end
        @test true
    end
    return
end
