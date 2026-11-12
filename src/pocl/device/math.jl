# `@fastmath` versions of math functions
#
# `Base.FastMath` falls back to the accurate functions for most of these, which
# SPIRVIntrinsics maps to the OpenCL builtins of the same name. Where PoCL has something
# measurably faster, the fast versions use that instead.
#
# PoCL vectorizes the accurate builtins with SLEEF; its `native_*` builtins are its own
# implementations, inlined into the kernel. Only the single-precision `native_sin`,
# `native_cos` and `native_tan` are faster: by 20-65% on x86-64 with AVX2, with errors of
# up to 2-3 ulp, but only for arguments up to about 10^4 in magnitude. Beyond that, their
# argument reduction is inaccurate.
#
# The other `native_*` and `half_*` builtins are either the accurate builtins (`native_log`,
# `native_exp`, `native_sqrt`, ..., and all of `half_*`), slower (the double-precision
# `native_sin`, `native_cos` and `native_tan`, and `native_powr`), or no faster than what
# `@fastmath` already emits (`native_divide`, `native_recip`, `native_rsqrt`). PoCL has no
# half-precision `native_sin` and `native_cos`.

# the mangled names of the builtins used here, which the compiler has to know are builtins.
# `@builtin_ccall` records them in SPIRVIntrinsics' `known_intrinsics` when it is expanded,
# but that modification of another package's array isn't kept when KernelAbstractions is
# precompiled.
const fastmath_builtins = String[]

for f in (:sin, :cos, :tan)
    builtin = "native_$f"
    push!(fastmath_builtins, "_Z$(length(builtin))$(builtin)$(mangle(Float32))")
    @eval @device_override Base.FastMath.$(Symbol(f, :_fast))(x::Float32) =
        @builtin_ccall($builtin, Float32, (Float32,), x)
end

# the `sincos` builtin returns the cosine through memory, which keeps PoCL from vectorizing
# it; separate calls to `sin` and `cos` are vectorized, and several times faster together.
# (the half-precision `sincos_fast` computes in single precision)
@device_override Base.FastMath.sincos_fast(x::Float32) =
    (Base.FastMath.sin_fast(x), Base.FastMath.cos_fast(x))
@device_override Base.FastMath.sincos_fast(x::Float64) = (sin(x), cos(x))
