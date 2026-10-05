# Precompile what the first kernel launch on the CPU back-end needs: the host code that
# launches a kernel, and above all the compiler that turns it into SPIR-V (GPUCompiler,
# LLVM.jl and the SPIR-V back-end), which otherwise takes many seconds to compile on first
# use. `@kernel` itself is covered by a workload in KernelAbstractions.jl.

@kernel function precompile_kernel(A, @Const(B))
    i = @index(Global, Linear)
    @inbounds A[i] = 2 * B[i] + 1
end

# on Julia 1.11, the launch leaks GPUCompiler's runtime into the package image, which then
# fails to link on Windows
const launch_in_workload = !(Sys.iswindows() && v"1.11-" <= VERSION < v"1.12-")

# whether the workload below launched its kernel, checked by the tests
const precompiled_launch = Ref(false)

PrecompileTools.@setup_workload begin
    if launch_in_workload && POCL.nanoOpenCL.pocl_standalone_jll.is_available() &&
            POCL.SPIRV_LLVM_Backend_jll.is_available() && POCL.SPIRV_Tools_jll.is_available()
        try
            # keep PoCL's kernel cache out of the user's cache directory, and don't have it
            # start a thread per core in every process that precompiles this package
            mktempdir() do cache_dir
                env = (
                    "POCL_CACHE_DIR" => cache_dir,
                    "POCL_CPU_MAX_CU_COUNT" => "1", "POCL_MAX_PTHREAD_COUNT" => "1",
                )
                withenv(env...) do
                    PrecompileTools.@compile_workload begin
                        A = Base.zeros(Float32, 4)
                        B = Base.ones(Float32, 4)
                        precompile_kernel(CPU())(A, B; ndrange = length(A))
                        synchronize(CPU())
                    end
                end
            end
            precompiled_launch[] = true
        catch err
            # a broken CPU back-end shouldn't keep this package from loading
            @debug "Failed to launch a kernel during precompilation" exception = (err, catch_backtrace())
        finally
            # don't serialize handles to OpenCL objects that only exist in this process
            POCL.reset_session_state!()
        end
    end
end
