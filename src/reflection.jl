import GPUCompiler

# GPUCompiler's `@device_code_*` macros install a compilation hook for the duration of the
# wrapped expression, so they report on every kernel that any GPUCompiler-based backend
# compiles while it runs -- the in-tree CPU backend as well as CUDA, AMDGPU, oneAPI or Metal.
#
# They are public but deliberately not exported, because the backend packages export macros of
# the same name and `using CUDA, KernelAbstractions` would make the name ambiguous.
for (macroname, stage) in (
        Symbol("@device_code_lowered") => "the lowered IR",
        Symbol("@device_code_typed") => "the type-inferred IR",
        Symbol("@device_code_warntype") => "the type-inferred IR, highlighting type instabilities",
        Symbol("@device_code_llvm") => "the generated LLVM IR",
        Symbol("@device_code_native") => "the generated machine code",
    )
    docstring = """
        KernelAbstractions.$macroname [kwargs...] ex

    Evaluate `ex` and, for every device kernel compiled along the way, show $stage.

    This is `GPUCompiler.$macroname`, re-exposed for convenience; see its documentation for the
    supported keyword arguments. It applies to any GPUCompiler-based backend, so wrapping a
    kernel launch works on the CPU backend and on GPU backends alike. Note that `ex` is really
    evaluated: the kernels it launches are compiled *and* run.

    # Examples

    ```julia
    KernelAbstractions.$macroname my_kernel(backend, 64)(A, ndrange=length(A))
    ```
    """
    @eval begin
        const $macroname = GPUCompiler.$macroname
        @doc $docstring $macroname
    end
end

"""
    KernelAbstractions.@device_code [dir=...] [...] ex

Evaluate `ex` and dump all forms of code generated for the device kernels it compiles to the
directory `dir`, or to a temporary directory if none is given.

This is `GPUCompiler.@device_code`, re-exposed for convenience; see its documentation for the
supported keyword arguments. Like the other `@device_code_*` macros it applies to any
GPUCompiler-based backend, and really evaluates `ex`.
"""
const var"@device_code" = GPUCompiler.var"@device_code"

# `public` is a parse error before 1.11
@static if VERSION >= v"1.11"
    eval(
        Meta.parse(
            """
            public @device_code_lowered, @device_code_typed, @device_code_warntype,
                @device_code_llvm, @device_code_native, @device_code
            """
        )
    )
end
