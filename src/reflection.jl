import InteractiveUtils
export @ka_code_typed

function ka_code_typed(kernel, argtypes; ndrange=nothing, workgroupsize=nothing)
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
    return InteractiveUtils.code_typed(KernelAbstractions.Cassette.overdub, (typeof(ctx), typeof(kernel.f), argtypes...))
end


macro ka_code_typed(ex0...)
    if length(ex0) == 1
        kw = ex0[1].args[end]
        kw.head = Symbol("=")
        resize!(ex0[1].args, length(ex0[1].args) - 1)
        ex0 = (kw, ex0[1])
    end

    thecall = InteractiveUtils.gen_call_with_extracted_types_and_kwargs(__module__, :ka_code_typed, ex0)

    quote
        local results = $thecall
        length(results) == 1 ? results[1] : results
    end
end
