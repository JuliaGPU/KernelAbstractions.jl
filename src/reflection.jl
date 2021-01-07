import InteractiveUtils
export @ka_code_typed, @ka_code_llvm

argconvert(k::Kernel{T}, arg) where T =
    error("Don't know how to convert arguments for Kernel{$T}")

using UUIDs
const Cthulhu = Base.PkgId(UUID("f68482b8-f384-11e8-15f7-abe071a5a75f"), "Cthulhu")

function ka_code_typed(kernel, argtypes; ndrange=nothing, workgroupsize=nothing, dependencies=nothing, interactive=false, kwargs...)
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
        mod===nothing && error("Interactive code reflection requires Cthulhu; please install and load this package first.")
        descend_code_typed = getfield(mod, :descend_code_typed)
        return descend_code_typed(KernelAbstractions.Cassette.overdub, (typeof(ctx), typeof(kernel.f), argtypes...); kwargs...)
    else
        return InteractiveUtils.code_typed(KernelAbstractions.Cassette.overdub, (typeof(ctx), typeof(kernel.f), argtypes...); kwargs...)
    end
end


function ka_code_llvm(kernel, argtypes; ndrange=nothing, workgroupsize=nothing, dependencies=nothing, kwargs...)
    ka_code_llvm(stdout, kernel, argtypes;  ndrange=ndrange, workgroupsize=nothing, dependencies=nothing, kwargs...)
end

function ka_code_llvm(io::IO, kernel, argtypes; ndrange=nothing, workgroupsize=nothing, dependencies=nothing, kwargs...)
    # get the iterspace and dynamic of a kernel
    ndrange, workgroupsize, iterspace, dynamic = KernelAbstractions.launch_config(kernel, ndrange, workgroupsize)

    # get the first block
    block = @inbounds KernelAbstractions.blocks(iterspace)[1]
    # get a context of the kernel based on the first block
    ctx = KernelAbstractions.mkcontext(kernel, block, ndrange, iterspace, dynamic)

    # reformat
    if argtypes isa Type
        argtypes = argtypes.parameters
    end
    # use code_typed
    return InteractiveUtils.code_llvm(io, KernelAbstractions.Cassette.overdub, (typeof(ctx), typeof(kernel.f), argtypes...); kwargs...)
end


function format_ex(ex0)
    ex = ()
    args = gensym(:args)
    old_args = nothing
    kern = nothing
    for i = 1:length(ex0)
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
                    kw.args[2] = esc(kw.args[2])
                    kw.head = Symbol("=")
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
Get the typed IR for a kernel

# Examples
```
@ka_code_typed kernel(args. ndrange=...)
@ka_code_typed kernel(args. ndrange=... workgroupsize=...)
@ka_code_typed optimize=false kernel(args. ndrange=...)
```
To use interactive mode (with Cthulhu), call
```
@ka_code_typed interactive=true kernel(args. ndrange=...)
```
If ndrange is statically defined, then you could call
```
@ka_code_typed kernel(args.)
```
Works for CPU or CUDA kernels, with static or dynamic declarations
"""
macro ka_code_typed(ex0...)
    ex, args, old_args, kern = format_ex(ex0)

    thecall = InteractiveUtils.gen_call_with_extracted_types_and_kwargs(__module__, :ka_code_typed, ex)

    quote
        local $(esc(args)) = $(old_args)
        # e.g. translate CuArray to CuDeviceArray
        $(esc(args)) = map(x->argconvert($kern, x), $(esc(args)))

        local results = $thecall
        if results !== nothing
            length(results) == 1 ? results[1] : results
        end
    end
end


"""
Get the llvm code for a kernel

# Examples
```
@ka_code_llvm kernel(args. ndrange=...)
@ka_code_llvm kernel(args. ndrange=... workgroupsize=...)
@ka_code_llvm optimize=false kernel(args. ndrange=...)
```
If ndrange is statically defined, then you could call
```
@ka_code_llvm kernel(args.)
```
Works for CPU kernels ONLY, with static or dynamic declarations
"""
macro ka_code_llvm(ex0...)
    ex, args, old_args, kern = format_ex(ex0)

    thecall = InteractiveUtils.gen_call_with_extracted_types_and_kwargs(__module__, :ka_code_llvm, ex)

    quote
        local $(esc(args)) = $(old_args)

        if isa($kern, Kernel{G} where {G<:GPU})
            # does not support GPU kernels
            error("@ka_code_llvm does not support GPU kernels")
        end

        local results = $thecall
    end
end
