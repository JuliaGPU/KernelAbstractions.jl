# `BFloat16` kernels on the POCL back end (#921). On Julia 1.11+, `BFloat16` is
# `Core.BFloat16`, which Julia lowers to LLVM's `bfloat`: storing it needs a PoCL that accepts
# `SPV_KHR_bfloat16`, and GPUCompiler widens its arithmetic to `float`. On Julia 1.10,
# BFloat16s.jl defines it as a 16-bit primitive type, which needs nothing from the device.

using BFloat16s

@kernel function bf16_copy!(y, @Const(x))
    i = @index(Global)
    @inbounds y[i] = x[i]
end
@kernel function bf16_axpy!(y, a, @Const(x))
    i = @index(Global)
    @inbounds y[i] = a * x[i] + y[i]
end
@kernel function bf16_ops!(out, @Const(x), @Const(y), @Const(z))
    i = @index(Global)
    @inbounds begin
        a, b, c = x[i], y[i], z[i]
        out[1, i] = a / b
        out[2, i] = a - b
        out[3, i] = min(a, b)
        out[4, i] = max(a, b)
        out[5, i] = abs(a)
        out[6, i] = copysign(a, b)
        out[7, i] = ifelse(a < b, a, b)
        out[8, i] = sqrt(abs(a))
        out[9, i] = floor(a)
        out[10, i] = BFloat16(isfinite(a / b))
        out[11, i] = -a
    end
end
@kernel function bf16_convert!(y, @Const(x))
    i = @index(Global)
    @inbounds y[i] = convert(eltype(y), x[i])
end

# the correctly rounded value, without relying on the host's `BFloat16` rounding: on Julia
# 1.11, vectorized ones crash LLVM on AVX512-BF16 hosts, and LLVM 19 to 22 (Julia 1.13) rounds
# ties wrongly in vectorized ones for AArch64 targets without BF16 (#949)
function bf16_nearest(x)
    b = BigFloat(x; precision = 256)
    # 8 significant bits, except for subnormals, which are multiples of 2^-133
    r = abs(b) < floatmin(Float32) ? round(b * big(2.0)^133) / big(2.0)^133 :
        BigFloat(b; precision = 8)
    return BFloat16(Float32(r))
end

const bf16_supported = !isdefined(Core, :BFloat16) ||
    "SPV_KHR_bfloat16" in POCL.device().spirv_extensions

if bf16_supported
    @testset "POCL BFloat16" begin
        backend = POCLBackend()
        n = 1024
        # finite values from bit patterns, rather than from rounding on the host (Julia 1.11's
        # vectorized `Float32` to `BFloat16` conversion crashes LLVM on AVX512-BF16 hosts)
        rand_bf16(n) = map(1:n) do _
            while true
                v = reinterpret(BFloat16, rand(UInt16))
                isfinite(v) && return v
            end
        end
        hx, hy, hz = rand_bf16(n), rand_bf16(n), rand_bf16(n)
        hx[1:4] .= BFloat16.((-0.0, 0.0, 1.0, -2.0))
        hy[1:4] .= BFloat16.((0.0, -0.0, 0.0, 3.0))
        x, y, z = (KernelAbstractions.allocate(backend, BFloat16, n) for _ in 1:3)
        copyto!(x, hx); copyto!(y, hy); copyto!(z, hz)

        @testset "copy" begin
            out = similar(x)
            bf16_copy!(backend)(out, x; ndrange = n)
            @test isequal(Array(out), hx)
        end

        @testset "axpy" begin
            out = copy(y)
            a = BFloat16(1.5)
            bf16_axpy!(backend)(out, a, x; ndrange = n)
            # widened to `float`, multiplied and added separately (each rounded to bfloat)
            @test isequal(Array(out), map((x, y) -> bf16_nearest(Float32(bf16_nearest(Float32(a) * Float32(x))) + Float32(y)), hx, hy))
        end

        @testset "arithmetic" begin
            out = KernelAbstractions.allocate(backend, BFloat16, 11, n)
            bf16_ops!(backend)(out, x, y, z; ndrange = n)
            o = Array(out)
            # (references in `Float32`: on Julia 1.11, the host miscompiles vectorized `BFloat16`
            # operations such as `copysign`)
            f32(op) = (xs...) -> bf16_nearest(op(Float32.(xs)...))
            @test isequal(o[1, :], map(f32(/), hx, hy))
            @test isequal(o[2, :], map(f32(-), hx, hy))
            @test isequal(o[3, :], map(f32(min), hx, hy))
            @test isequal(o[4, :], map(f32(max), hx, hy))
            @test isequal(o[5, :], map(f32(abs), hx))
            @test isequal(o[6, :], map(f32(copysign), hx, hy))
            @test isequal(o[7, :], map(f32((a, b) -> ifelse(a < b, a, b)), hx, hy))
            @test isequal(o[8, :], map(f32(a -> sqrt(abs(a))), hx))
            @test isequal(o[9, :], map(f32(floor), hx))
            @test isequal(o[10, :], map(f32((a, b) -> isfinite(a / b)), hx, hy))
            @test isequal(o[11, :], map(f32(-), hx))
        end

        # including inputs that a conversion through `Float32` would round wrongly
        @testset "$T to BFloat16" for (T, extra) in (
                (Float32, Float32[1 + 2.0f0^-8 + 2.0f0^-20, floatmax(Float32)]),
                (Float64, [1 + 2.0^-8 + 2.0^-30, -(1 + 2.0^-8 + 2.0^-30), 1.0e39]),
                (Int32, Int32[2^30 + 2^22 + 1, typemin(Int32), typemax(Int32)]),
                (Int64, Int64[2^62 + 2^54 + 1, typemin(Int64), typemax(Int64)]),
                (UInt64, [UInt64(2)^63 + UInt64(2)^55 + 1, typemax(UInt64)]),
            )
            vals = T <: AbstractFloat ? T.(randn(n) .* 1000) : rand(T, n)
            vals[1:length(extra)] .= extra
            src = KernelAbstractions.allocate(backend, T, n)
            copyto!(src, vals)
            dst = KernelAbstractions.allocate(backend, BFloat16, n)
            bf16_convert!(backend)(dst, src; ndrange = n)
            @test isequal(Array(dst), map(bf16_nearest, vals))
        end

        @testset "BFloat16 to $T" for T in (Float32, Float64)
            dst = KernelAbstractions.allocate(backend, T, n)
            bf16_convert!(backend)(dst, x; ndrange = n)
            @test isequal(Array(dst), map(T, hx))
        end
    end
else
    # PoCL without `SPV_KHR_bfloat16` (pocl_standalone_jll before 7.2.3+3): compiling must
    # fail cleanly, rather than PoCL aborting on a rejected extension
    @testset "POCL BFloat16 (unsupported)" begin
        x = BFloat16.(Float32[1, 2, 3, 4])
        @test_throws POCL.GPUCompiler.InvalidIRError bf16_copy!(POCLBackend())(similar(x), x; ndrange = 4)
    end
end
