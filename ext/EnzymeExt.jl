module EnzymeExt
    using EnzymeCore
    using EnzymeCore.EnzymeRules
    import KernelAbstractions: Kernel, StaticSize, launch_config, __groupsize, __groupindex, blocks, mkcontext, CompilerMetadata

    EnzymeRules.inactive(::typeof(StaticSize), x...) = nothing

    function first_type(::Type{NamedTuple{A, B}}) where {A,B}
        first_type(B)
    end
    function first_type(::Type{T}) where {T<:Tuple}
        T.parameters[1]
    end

    function fwd(ctx, forward, f, args...)
        forward(Const(f), Const(ctx), args...)
        return nothing
    end

    function aug_fwd(ctx, forward, f, subtape, args...)
        subtape[__groupindex(ctx)] = forward(Const(f), Const(ctx), args...)[1]
        return nothing
    end

    function rev(ctx, reverse, f, subtape, args...)
        tp = subtape[__groupindex(ctx)]
        reverse(Const(f), Const(ctx), args..., tp)
        return nothing
    end

    function EnzymeRules.forward(func::Const{<:Kernel}, ::Type{Const{Nothing}}, args...; ndrange=nothing, workgroupsize=nothing)
        kernel = func.val
        f = kernel.f

        ndrange, workgroupsize, iterspace, dynamic = launch_config(kernel, ndrange, workgroupsize)
        # ctxTy = CompilerMetadata{ndrange(kernel), Core.Typeof(dynamic)}
        block = first(blocks(iterspace))
        ctx = mkcontext(kernel, block, ndrange, iterspace, dynamic)
        ctxTy = Core.Typeof(ctx) # CompilerMetadata{ndrange(kernel), Core.Typeof(dynamic)}
        tt′ = Tuple{Const{ctxTy}, map(Core.Typeof, args)...}
        FT = Const{Core.Typeof(f)}

        forward = EnzymeCore.autodiff_thunk(Forward, FT, Const, tt′.parameters...)
        fwd_kernel = similar(kernel, fwd)

        fwd_kernel(forward, f, args...; ndrange, workgroupsize)
    end

    function EnzymeRules.augmented_primal(config::Config, func::Const{<:Kernel}, ::Type{Const{Nothing}}, args...; ndrange=nothing, workgroupsize=nothing)
        kernel = func.val
        f = kernel.f

        ndrange, workgroupsize, iterspace, dynamic = launch_config(kernel, ndrange, workgroupsize)
        block = first(blocks(iterspace))


        ctx = mkcontext(kernel, block, ndrange, iterspace, dynamic)
        ctxTy = Core.Typeof(ctx) # CompilerMetadata{ndrange(kernel), Core.Typeof(dynamic)}
        tt′ = Tuple{Const{ctxTy}, map(Core.Typeof, args)...}

        # TODO autodiff_deferred on the func.val
        ModifiedBetween = Val((overwritten(config)[1], false, overwritten(config)[2:end]...))

        FT = Const{Core.Typeof(f)}

        forward, reverse = EnzymeCore.autodiff_thunk(ReverseSplitModified(ReverseSplitWithPrimal, ModifiedBetween), FT, Const,  tt′.parameters...)

        rt = Core.Compiler.return_type(forward, Tuple{FT, tt′.parameters...})
        TapeType = first_type(rt)

        subtape = Array{TapeType}(undef, __groupsize(ctx))

        aug_kernel = similar(kernel, aug_fwd)

        aug_kernel(forward, f, subtape, args...; ndrange, workgroupsize)

        # TODO the fact that ctxTy is type unstable means this is all type unstable.
        # Since custom rules require a fixed return type, explicitly cast to Any, rather
        # than returning a AugmentedReturn{Nothing, Nothing, T} where T.
        tape = (reverse, subtape)::Any

        res =  AugmentedReturn{Nothing, Nothing, Tuple{Any, Vector}}(nothing, nothing, tape)

        return res
    end

    function EnzymeRules.reverse(::Config, func::Const{<:Kernel}, ::Type{<:EnzymeCore.Annotation}, tape, args...; ndrange=nothing, workgroupsize=nothing)
        reverse, subtape = tape
        rev_kernel = similar(func.val, rev)
        rev_kernel(reverse, func.val.f, subtape, args...; ndrange, workgroupsize)
        return ()
    end
end
