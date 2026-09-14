import InteractiveUtils
import GPUCompiler
export @ka_code_typed

using UUIDs
const Cthulhu = Base.PkgId(UUID("f68482b8-f384-11e8-15f7-abe071a5a75f"), "Cthulhu")

function ka_code_typed(kernel, argtypes; ndrange = nothing, workgroupsize = nothing, interactive = false, kwargs...)
    # get the iterspace and dynamic of a kernel
    ndrange, workgroupsize, iterspace, dynamic = KernelAbstractions.launch_config(kernel, ndrange, workgroupsize)

    if isa(kernel, Kernel{CPU})
        # get the first block
        block = @inbounds KernelAbstractions.blocks(iterspace)[1]
        # get a context of the kernel based on the first block
        ctx = KernelAbstractions.mkcontext(kernel, block, ndrange, iterspace, dynamic)
    else
        ctx = KernelAbstractions.mkcontext(kernel, ndrange, iterspace)
    end
    # reformat
    if argtypes isa Type
        argtypes = argtypes.parameters
    end
    # use code_typed
    if interactive
        # call Cthulhu without introducing a dependency on Cthulhu
        mod = Base.get(Base.loaded_modules, Cthulhu, nothing)
        mod === nothing && error("Interactive code reflection requires Cthulhu; please install and load this package first.")
        descend_code_typed = getfield(mod, :descend_code_typed)
        return descend_code_typed(typeof(kernel.f), (typeof(ctx), argtypes...); kwargs...)
    else
        return InteractiveUtils.code_typed(kernel.f, (typeof(ctx), argtypes...); kwargs...)
    end
end


function format_ex(ex0)
    ex = ()
    args = gensym(:args)
    old_args = nothing
    kern = nothing
    for i in 1:length(ex0)
        if ex0[i].head == :call
            # inside kernel() expr
            while length(ex0[i].args) > 2
                if isa(ex0[i].args[end], Expr)
                    # at expr (like ndrange=10)
                    kw = ex0[i].args[end]
                    if kw.head != :kw
                        # if an expr in place of a variable, skip
                        break
                    end
                    # see https://github.com/JuliaLang/julia/pull/41040
                    @static VERSION < v"1.7.0-DEV.1221" && (kw.args[2] = esc(kw.args[2]))
                    kw.head = :(=)
                    resize!(ex0[i].args, length(ex0[i].args) - 1)
                    ex = (kw,)..., ex...
                else
                    # only symbols left
                    break
                end
            end
            # save kernel args
            old_args = Expr(:tuple, map(esc, ex0[i].args[2:end])...)
            resize!(ex0[i].args, 2)
            ex0[i].args[2] = Expr(:..., args)
            kern = esc(ex0[i].args[1])
        end
        ex = ex..., ex0[i]
    end
    @assert(old_args != nothing)
    @assert(kern != nothing)
    return ex, args, old_args, kern
end


"""
    @ka_code_typed [kwargs...] kernel(args...; ndrange=..., workgroupsize=...)

Return the typed IR for a kernel's device function, similar to `InteractiveUtils.code_typed`.

Pass `interactive=true` to descend into the IR with [Cthulhu](https://github.com/JuliaDebug/Cthulhu.jl)
(must be loaded in the session). If `ndrange` is fixed at kernel construction time, it can be
omitted at the call site.

This reflects on the kernel function as the host sees it, which is independent of the backend.
To inspect the code a backend actually generates, use [`KernelAbstractions.@device_code_typed`](@ref)
or one of the other `@device_code_*` macros.

# Examples

```julia
@ka_code_typed my_kernel(backend)(A, ndrange=length(A))
@ka_code_typed my_kernel(backend, 64)(A, ndrange=length(A))
@ka_code_typed optimize=false my_kernel(backend)(A, ndrange=length(A))
@ka_code_typed interactive=true my_kernel(CPU())(A, ndrange=length(A))
```
"""
macro ka_code_typed(ex0...)
    ex, args, old_args, kern = format_ex(ex0)

    thecall = InteractiveUtils.gen_call_with_extracted_types_and_kwargs(__module__, :ka_code_typed, ex)

    return quote
        local $(esc(args)) = $(old_args)
        # e.g. translate CuArray to CuBackendArray
        $(esc(args)) = map(x -> argconvert($kern, x), $(esc(args)))

        local results = $thecall
        if results !== nothing
            length(results) == 1 ? results[1] : results
        end
    end
end


#
# Device code reflection
#

# GPUCompiler's `@device_code_*` macros install a compilation hook for the duration of the
# wrapped expression, so they report on every kernel that any GPUCompiler-based backend
# compiles while it runs -- the in-tree CPU backend as well as CUDA, AMDGPU, oneAPI or Metal.
# That makes them the backend-agnostic way to inspect generated device code, which the
# host-side `@ka_code_typed` cannot be: it reflects on the kernel function as the host sees
# it, before the backend's compilation pipeline has run.
#
# They are forwarded here rather than exported, because the backend packages export macros of
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
