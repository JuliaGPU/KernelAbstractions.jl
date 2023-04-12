module EnzymeExt
    using EnzymeCore
    using EnzymeCore.EnzymeRules
    import KernelAbstractions: Kernel, StaticSize, launch_config, __groupsize, __groupindex, blocks, mkcontext

    EnzymeRules.inactive(::typeof(StaticSize), x...) = nothing

    function first_type(::Type{NamedTuple{A, B}}) where {A,B}
        first_type(B)
    end
    function first_type(::Type{T}) where {T<:Tuple}
        T.parameters[1]
    end

    function EnzymeRules.augmented_primal(config::Config, func::Const{<:Kernel}, ::Nothing, args...; ndrange = nothing, workgroupsize = nothing)
        # Pre-allocate tape according to ndrange... * inner_tape
        kernel = func.val
        f = kernel.f

        ndrange, workgroupsize, iterspace, dynamic = launch_config(kernel, ndrange, workgroupsize)
        block = first(blocks(iterspace))

        ctx = mkcontext(kernel, block, ndrange, iterspace, dynamic)
        ctxTy = typeof(ctx) # Base._return_type(mkcontext, Tuple{typeof(kernel), typeof(block), typeof(ndrange), typeof(iterspace), typeof(dynamic)})
        tt′ = Tuple{Const{ctxTy}, map(Core.typeof, args)...}

        # TODO autodiff_deferred on the func.val
        ModifiedBetween = Val((overwritten(config)[1], false, overwritten(config)[2:end]...))

        forward, pullback0 = Enzyme.autodiff_thunk(ReverseSplitModified(ReverseSplitWithPrimal, ModifiedBetween), Const{Core.Typeof(f)}, Duplicated,  tt′)

        TapeType = first_type(Base._return_type(forward, tt′))

        subtape = Array{TapeType}(undef, __groupsize(ctx))

        function fwd(ctx, subtape, args...)
            subtape[__groupindex(ctx)] = forward(Const(ctx), args...)[1]
            return nothing
        end

        function rev(ctx, subtape, args...)
            reverse(Const(ctx), args..., subtape[__groupindex(ctx)])
            return nothing
        end

        aug_kernel = similar(kernel, fwd)
        rev_kernel = similar(kernel, rev)

        aug_kernel(subtape, args...; ndrange, workgroupsize)

        tape = (; rev_kernel, subtape)

        return AugmentedReturn(nothing, nothing, tape)
    end

    function EnzymeRules.reverse(::Config, ::Const{<:Kernel}, ::Type{<:EnzymeCore.Annotation}, tape, args...; ndrange=nothing, workgroupsize=nothing)
        rev_kernel, subtape = tape
        rev_kernel(subtape, args...; ndrange, workgroupsize)
        return ()
    end
end
