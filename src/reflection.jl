import InteractiveUtils
export @ka_code_typed
using CUDA

function ka_code_typed(kernel, argtypes; ndrange=nothing, workgroupsize=nothing, dependencies=nothing, kwargs...)
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
    return InteractiveUtils.code_typed(KernelAbstractions.Cassette.overdub, (typeof(ctx), typeof(kernel.f), argtypes...); kwargs...)
end


"""
Get the llvm code for a kernel

# Examples
```
@ka_code_typed kernel(args)
@ka_code_typed optimize=false kernel(args)
```
Works for CPU or CUDA kernels, with static or dynamic declarations
"""
macro ka_code_typed(ex0...)
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
                    @assert kw.head == :kw
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

    thecall = InteractiveUtils.gen_call_with_extracted_types_and_kwargs(__module__, :ka_code_typed, ex)

    quote
        local $(esc(args)) = $(old_args)
        if isa($kern, Kernel{CUDADevice})
            # translate CuArray to CuDeviceArray
            local $(esc(args)) = map(CUDA.cudaconvert, $(esc(args)))
        end

        local results = $thecall
        length(results) == 1 ? results[1] : results
    end
end
