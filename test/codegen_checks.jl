# Assertions about the device code KernelAbstractions generates, written as LLVM FileCheck
# directives over `@device_code_llvm` output.
#
# codegen.jl runs this file in a subprocess rather than including it, because `Pkg.test`
# defaults to `--check-bounds=yes`, which forces bounds checks on regardless of
# `@inbounds` and so distorts every kernel here. Run it directly to iterate on a pattern:
#
#     julia --project=test test/codegen_checks.jl
#
# These deliberately live outside `Testsuite`. That module is included by the backend
# packages, which cannot pick up new test dependencies (see the note at the top of
# testsuite.jl), and the patterns below are specific to the in-tree POCL/SPIR-V back-end
# anyway. Patterns avoid pointer syntax, which differs between the typed pointers of
# LLVM 15 (Julia 1.10) and the opaque pointers of later versions.

using FileCheck
using KernelAbstractions
using KernelAbstractions: @atomic
using StaticArrays
using Test

import KernelAbstractions.POCL: @device_code_llvm

@kernel function codegen_mul2(A)
    I = @index(Global, Linear)
    A[I] = 2 * A[I]
end

@kernel function codegen_mul2_inbounds(A)
    I = @index(Global, Linear)
    @inbounds A[I] = 2 * A[I]
end

@kernel function codegen_reverse(A)
    N = @uniform prod(@groupsize())
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    lmem = @localmem Float32 (N,)
    @inbounds begin
        lmem[i] = A[I]
        @synchronize
        A[I] = lmem[N - i + 1]
    end
end

@kernel function codegen_atomic_sum(A, out)
    I = @index(Global, Linear)
    @inbounds @atomic out[1] += A[I]
end

@kernel function codegen_print()
    I = @index(Global, Linear)
    @print("index ", I, "\n")
end

@kernel function codegen_private_reduce(A)
    I = @index(Global, Linear)
    priv = @private Float32 (8,)
    for j in 1:8
        @inbounds priv[j] = A[I] * j
    end
    @inbounds A[I] = sum(priv) + maximum(priv) + foldl(-, priv)
end

@noinline private_consume(priv) = @inbounds priv[1] + priv[8]

@kernel function codegen_private_escape(A)
    I = @index(Global, Linear)
    priv = @private Float32 (8,)
    for j in 1:8
        @inbounds priv[j] = A[I] * j
    end
    @inbounds A[I] = private_consume(priv)
end

@kernel function codegen_global_linear(A)
    I = @index(Global, Linear)
    @inbounds A[I] = I
end

# `@inbounds` is only honoured under `--check-bounds=auto`; several checks below assert
# that it removes code, so refuse to run under anything else rather than fail obscurely.
if Base.JLOptions().check_bounds != 0
    error("codegen_checks.jl requires --check-bounds=auto")
end

@testset "Codegen" begin
    backend = POCLBackend()
    A = KernelAbstractions.zeros(backend, Float32, 64)
    out = KernelAbstractions.zeros(backend, Float32, 1)

    # The global index is computed from the SPIR-V work-item builtins, with no call back
    # into the Julia runtime, and array accesses go to the global address space.
    @testset "index computation" begin
        @test @filecheck implicit_check_not = "jl_" begin
            @check "define spir_kernel void @{{.*}}gpu_codegen_mul2_inbounds"
            @check "__spirv_BuiltInWorkgroupId"
            @check "__spirv_BuiltInLocalInvocationId"
            @check "load float, {{.*}}addrspace(1)"
            @check "store float {{.*}}addrspace(1)"
            @check "ret void"
            @device_code_llvm debuginfo = :none codegen_mul2_inbounds(backend)(A, ndrange = 64)
            KernelAbstractions.synchronize(backend)
        end
    end

    # Without `@inbounds` the kernel carries the out-of-bounds path: a printf of the error
    # and the GPUCompiler exception signalling. `@inbounds` must remove all of it.
    @testset "bounds checks" begin
        @test @filecheck begin
            @check "define spir_kernel void @{{.*}}gpu_codegen_mul2"
            @check "@printf"
            @check "gpu_report_exception"
            @check "gpu_signal_exception"
            @device_code_llvm debuginfo = :none codegen_mul2(backend)(A, ndrange = 64)
            KernelAbstractions.synchronize(backend)
        end

        @test @filecheck implicit_check_not = ["gpu_report_exception", "@printf"] begin
            @check "define spir_kernel void @{{.*}}gpu_codegen_mul2_inbounds"
            @check "ret void"
            @device_code_llvm debuginfo = :none codegen_mul2_inbounds(backend)(A, ndrange = 64)
            KernelAbstractions.synchronize(backend)
        end
    end

    # With both the workgroupsize and the ndrange known at compile time, `__validindex`
    # folds away: nothing is left to branch on, so the kernel body is straight-line code.
    @testset "static ndrange" begin
        @test @filecheck implicit_check_not = "br i1" begin
            @check "define spir_kernel void @{{.*}}gpu_codegen_mul2_inbounds"
            @check "ret void"
            @device_code_llvm debuginfo = :none codegen_mul2_inbounds(backend, 16, 64)(A)
            KernelAbstractions.synchronize(backend)
        end

        # a dynamic ndrange keeps the check, so the test above is not vacuous
        @test @filecheck begin
            @check "define spir_kernel void @{{.*}}gpu_codegen_mul2_inbounds"
            @check "br i1"
            @device_code_llvm debuginfo = :none codegen_mul2_inbounds(backend, 16)(A, ndrange = 64)
            KernelAbstractions.synchronize(backend)
        end
    end

    # `@localmem` becomes a module-level allocation in the SPIR-V workgroup address space
    # (3), which the kernel reads and writes directly. The two accesses are `@check_dag`
    # because LLVM is free to emit the basic blocks in any order.
    @testset "localmem" begin
        @test @filecheck begin
            @check "@local_memory = {{.*}}addrspace(3) global"
            @check "define spir_kernel void @{{.*}}gpu_codegen_reverse"
            @check_dag "store float {{.*}}addrspace(3)"
            @check_dag "load float, {{.*}}addrspace(3)"
            @device_code_llvm debuginfo = :none dump_module = true codegen_reverse(backend, 16)(A, ndrange = 64)
            KernelAbstractions.synchronize(backend)
        end
    end

    # `@synchronize` becomes a SPIR-V control barrier.
    @testset "synchronize" begin
        @test @filecheck begin
            @check "define spir_kernel void @{{.*}}gpu_codegen_reverse"
            @check "call {{.*}}@{{.*}}__spirv_ControlBarrier"
            @device_code_llvm debuginfo = :none codegen_reverse(backend, 16)(A, ndrange = 64)
            KernelAbstractions.synchronize(backend)
        end
    end

    # `@atomic` lowers to a native floating-point addition on global memory, which PoCL
    # supports, rather than to a lock or a compare-and-swap loop.
    @testset "atomics" begin
        @test @filecheck implicit_check_not = "{{cmpxchg|AtomicCompareExchange}}" begin
            @check "define spir_kernel void @{{.*}}gpu_codegen_atomic_sum"
            @check "call float @{{.*}}__spirv_AtomicFAddEXT"
            @device_code_llvm debuginfo = :none codegen_atomic_sum(backend, 16)(A, out, ndrange = 64)
            KernelAbstractions.synchronize(backend)
        end
    end

    # An N-d launch maps the work-item builtins onto a dynamic 3-D iteration space directly,
    # without decomposing linear ids.
    @testset "N-d launch" begin
        B = KernelAbstractions.zeros(backend, Int, 4, 5, 6)
        @test @filecheck implicit_check_not = "{{[us]div i(32|64)}}" begin
            @check "define spir_kernel void @{{.*}}gpu_codegen_global_linear"
            @check "ret void"
            @device_code_llvm debuginfo = :none codegen_global_linear(backend)(B, ndrange = size(B))
            KernelAbstractions.synchronize(backend)
        end

        # a linear launch does, so the test above is not vacuous
        kernel = codegen_global_linear(backend)
        ndrange, workgroupsize, iterspace, _ = KernelAbstractions.launch_config(kernel, size(B), nothing)
        @test @filecheck begin
            @check "define spir_kernel void @{{.*}}gpu_codegen_global_linear"
            @check "udiv i32"
            @device_code_llvm debuginfo = :none KernelAbstractions.launch_kernel(
                kernel, KernelAbstractions.LinearLaunch{Int32}(), ndrange, workgroupsize, iterspace, (B,)
            )
            KernelAbstractions.synchronize(backend)
        end
    end

    # With StaticArrays loaded, whole-array reductions over `@private` storage are unrolled,
    # so the stack slot is promoted to registers rather than read through memory.
    @testset "private" begin
        @test @filecheck implicit_check_not = "alloca" begin
            @check "define spir_kernel void @{{.*}}gpu_codegen_private_reduce"
            @check "ret void"
            @device_code_llvm debuginfo = :none codegen_private_reduce(backend, 16)(A, ndrange = 64)
            KernelAbstractions.synchronize(backend)
        end

        # storage passed to a function that isn't inlined stays on the stack, so the test
        # above is not vacuous
        @test @filecheck begin
            @check "define spir_kernel void @{{.*}}gpu_codegen_private_escape"
            @check "alloca"
            @device_code_llvm debuginfo = :none codegen_private_escape(backend, 16)(A, ndrange = 64)
            KernelAbstractions.synchronize(backend)
        end
    end

    # `@print` lowers to a single variadic printf call, not to one call per argument.
    @testset "print" begin
        @test @filecheck begin
            @check "define spir_kernel void @{{.*}}gpu_codegen_print"
            @check "@printf"
            @check_not "@printf"
            @device_code_llvm debuginfo = :none codegen_print(backend, 16)(ndrange = 16)
            KernelAbstractions.synchronize(backend)
        end
    end
end
