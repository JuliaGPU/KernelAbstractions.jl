import InteractiveUtils
export @ka_code_typed

function ka_code_typed(kernel, argtypes; ndrange=nothing, workgroupsize=nothing, dependencies=nothing, kwargs...)
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
    return InteractiveUtils.code_typed(KernelAbstractions.Cassette.overdub, (typeof(ctx), typeof(kernel.f), argtypes...); kwargs...)
end


macro ka_code_typed(ex0...)
    ex = ()
    for i = 1:length(ex0)
        if ex0[i].head == :call
            while length(ex0[i].args) > 2
                kw = ex0[i].args[end]
                @assert kw.head == :kw
                kw.args[2] = esc(kw.args[2])
                kw.head = Symbol("=")
                resize!(ex0[i].args, length(ex0[i].args) - 1)
                ex = (kw,)..., ex...
            end
        end
        ex = ex..., ex0[i]
    end

    thecall = InteractiveUtils.gen_call_with_extracted_types_and_kwargs(__module__, :ka_code_typed, ex)

    quote
        local results = $thecall
        length(results) == 1 ? results[1] : results
    end
end
