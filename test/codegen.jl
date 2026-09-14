# Codegen tests: assertions about the device code KernelAbstractions generates, written as
# LLVM FileCheck directives over `@device_code_llvm` output.
#
# These deliberately live outside `Testsuite`. That module is included by the backend
# packages, which cannot pick up new test dependencies (see the note at the top of
# testsuite.jl), and the patterns below are specific to the in-tree POCL/SPIR-V back-end
# anyway. Patterns avoid pointer syntax, which differs between the typed pointers of
# LLVM 15 (Julia 1.10) and the opaque pointers of later versions.

using FileCheck
using KernelAbstractions
using KernelAbstractions: @atomic
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

function codegen_testsuite(backend)
    A = KernelAbstractions.zeros(backend, Float32, 64)
    out = KernelAbstractions.zeros(backend, Float32, 1)

    # The global index is computed from the SPIR-V work-item builtins, with no call back
    # into the Julia runtime, and array accesses go to the global address space.
    @testset "index computation" begin
        @test @filecheck implicit_check_not = "jl_" begin
            @check "define spir_kernel void @{{.*}}gpu_codegen_mul2_inbounds"
            @check "@__spirv_BuiltInWorkgroupId"
            @check "@__spirv_BuiltInLocalInvocationId"
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

    # `@atomic` lowers to a native atomicrmw on global memory rather than to a lock or a
    # compare-and-swap loop.
    @testset "atomics" begin
        @test @filecheck implicit_check_not = "cmpxchg" begin
            @check "define spir_kernel void @{{.*}}gpu_codegen_atomic_sum"
            @check "atomicrmw fadd"
            @device_code_llvm debuginfo = :none codegen_atomic_sum(backend, 16)(A, out, ndrange = 64)
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

    return
end
