using KernelAbstractions
using Random
using Test

include("quality_assurance.jl")
include("testsuite.jl")

@testset "Quality assurance" begin
    quality_assurance_testsuite()
end

KernelAbstractions.versioninfo(POCLBackend())
@info "Configuration" pocl = KernelAbstractions.POCL.nanoOpenCL.pocl_standalone_jll.libpocl

import KernelAbstractions.POCL: POCL, @opencl, @device_code_llvm

@testset "POCL float atomics" begin
    # pocl's CPU device natively supports float add and min/max atomics in both global
    # and local memory, so the SPIR-V extensions guarding them must be permitted
    dev = POCL.device()
    exts = split(POCL.default_spirv_extensions(dev), ",")
    @test "+SPV_EXT_shader_atomic_float_add" in exts
    @test "+SPV_EXT_shader_atomic_float_min_max" in exts
    @test dev.half_fp_atomic_capabilities == 0
    # an explicit list overrides the device-derived default
    config = POCL.compiler_config(dev; extensions = "+SPV_KHR_expect_assume")
    @test config.target.extensions == "+SPV_KHR_expect_assume"
end

# `randn`/`randexp` for Float16 route through Random's table-free fallback, whose polar
# transform overflows in Float16 and whose `log1p` isn't available for Float16 on the
# device. The device overlays compute in Float32 and convert, so results stay finite.
if "cl_khr_fp16" in POCL.device().extensions
    @testset "POCL device RNG: Float16" begin
        @kernel function f16_rng_kernel(A, B)
            i = @index(Global, Linear)
            @inbounds A[i] = Random.randn(Float16)
            @inbounds B[i] = Random.randexp(Float16)
        end

        # the overflow this guards against hits a few hundred values in 2^20 draws,
        # so a small sample would not catch a regression
        len = 2^20
        a = KernelAbstractions.zeros(POCLBackend(), Float16, len)
        b = KernelAbstractions.zeros(POCLBackend(), Float16, len)
        f16_rng_kernel(POCLBackend())(a, b; ndrange = len)
        KernelAbstractions.synchronize(POCLBackend())
        @test all(isfinite, a)
        @test all(isfinite, b)
    end
end

@testset "POCL compilation cache" begin
    mod = @eval module $(gensym())
    @noinline child() = return
    kernel() = child()
    end

    count() = POCL.compilations[]
    launch() = @opencl mod.kernel()

    # the initial launch compiles
    n = count()
    Base.invokelatest(launch)
    @test count() == n + 1

    # a second launch hits the cache
    Base.invokelatest(launch)
    @test count() == n + 1

    # jobs differing only in codegen-level settings get their own artifacts...
    POCL.clfunction(mod.kernel, Tuple{}; name = "custom")
    @test count() == n + 2
    # ... which are cached as well
    POCL.clfunction(mod.kernel, Tuple{}; name = "custom")
    @test count() == n + 2

    # reflection observes already-compiled kernels (by forcing recompilation,
    # which must leave the cached entry in a usable state)
    @test !isempty(sprint(io -> (@device_code_llvm io = io Base.invokelatest(launch))))
    n = count()
    Base.invokelatest(launch)
    @test count() == n

    # redefining the kernel recompiles...
    @eval mod kernel() = (child(); child())
    Base.invokelatest(launch)
    @test count() == n + 1
    # ... as does redefining a callee
    @eval mod @noinline child() = nothing
    Base.invokelatest(launch)
    @test count() == n + 2
end

@testset "POCL debug level" begin
    # `debug_level` selects how much exception-reporting code a kernel carries,
    # independent of the session's `-g` level
    oob(a) = (a[2] = 1.0f0; return)
    a = zeros(Float32, 1)
    ir(dl) = sprint(io -> (@device_code_llvm io = io @opencl launch = false debug_level = dl oob(a)))
    @test !occursin("gpu_report_exception", ir(0))
    @test occursin("gpu_report_exception", ir(1))
    @test !occursin("gpu_report_exception_frame", ir(1))
    @test occursin("gpu_report_exception_frame", ir(2))
end

@testset "POCL device-side exceptions" begin
    # a kernel that throws must not wedge the device: it should complete, and surface on
    # the host as a `KernelException`. POCL launches synchronously, so the exception is
    # reported by the launch itself.
    @kernel function throwing_kernel(a)
        i = @index(Global, Linear)
        a[i + 1] = 1.0f0      # out-of-bounds store on a length-1 array
    end
    @kernel function fill_one(a)
        i = @index(Global, Linear)
        @inbounds a[i] = 1.0f0
    end

    a = KernelAbstractions.zeros(POCLBackend(), Float32, 1)
    @test_throws POCL.KernelException throwing_kernel(POCLBackend())(a; ndrange = 1)

    # the mailbox is reset on read, so the device stays usable
    KernelAbstractions.synchronize(POCLBackend())
    b = KernelAbstractions.zeros(POCLBackend(), Float32, 4)
    fill_one(POCLBackend())(b; ndrange = 4)
    KernelAbstractions.synchronize(POCLBackend())
    @test b == ones(Float32, 4)

    # `@opencl` does not wait for its event, so the report is consumed by the next check
    oob(a) = (a[2] = 1.0f0; return)
    @opencl oob(a)
    @test_throws POCL.KernelException POCL.check_exceptions()
    POCL.check_exceptions()

    # an exception whose argument needs boxing (a runtime value) must not have its throw
    # path deleted by the device compiler (JuliaGPU/GPUCompiler.jl#919)
    function boxed(a)
        x = a[1]
        x == 0 && throw(DomainError(x))
        return
    end
    @test occursin(
        "gpu_signal_exception",
        sprint(io -> (@device_code_llvm io = io @opencl launch = false boxed(a)))
    )
    @opencl boxed(a)
    @test_throws POCL.KernelException POCL.check_exceptions()

    # a quirk records its own name and reason, from debug level 1 on
    exc = try
        throwing_kernel(POCLBackend())(a; ndrange = 1)
        nothing
    catch err
        err
    end
    @test exc isa POCL.KernelException
    if Base.JLOptions().debug_level >= 1
        @test exc.name == "BoundsError"
        @test exc.reason == "Out-of-bounds array access"
        @test occursin("BoundsError", sprint(showerror, exc))
    end
end

@testset "CPU back-end" begin
    struct CPUBackendArray{T, N, A} end # Fake and unused
    Testsuite.testsuite(CPU, "CPU", Base, Array, CPUBackendArray)
end

struct NewBackend <: KernelAbstractions.GPU end
@testset "Default host implementation" begin
    backend = NewBackend()

    @test_throws MethodError KernelAbstractions.synchronize(backend)

    @test_throws MethodError KernelAbstractions.allocate(backend, Float32, 1)
    @test_throws MethodError KernelAbstractions.allocate(backend, Float32, (1,))
    @test_throws MethodError KernelAbstractions.allocate(backend, Float32, 1, 2)

    @test_throws MethodError KernelAbstractions.zeros(backend, Float32, 1)
    @test_throws MethodError KernelAbstractions.ones(backend, Float32, 1)

    @test KernelAbstractions.supports_atomics(backend) == true
    @test KernelAbstractions.supports_float64(backend) == true

    @test KernelAbstractions.priority!(backend, :high) === nothing
    @test KernelAbstractions.priority!(backend, :normal) === nothing
    @test KernelAbstractions.priority!(backend, :low) === nothing

    @test_throws ErrorException KernelAbstractions.priority!(backend, :middle)

    @test KernelAbstractions.functional(backend) === missing
end


# include("extensions/enzyme.jl")
# @static if VERSION >= v"1.7.0"
#     @testset "Enzyme" begin
#         enzyme_testsuite(CPU, Array)
#     end
# end
