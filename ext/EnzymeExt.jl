module EnzymeExt
    if isdefined(Base, :get_extension)
        using EnzymeCore
        using EnzymeCore.EnzymeRules
    else
        using ..EnzymeCore
        using ..EnzymeCore.EnzymeRules
    end
    import KernelAbstractions: Kernel, StaticSize, launch_config, __groupsize, __groupindex, blocks, mkcontext, CompilerMetadata, CPU

    EnzymeRules.inactive(::Type{StaticSize}, x...) = nothing

    function fwd(ctx, f, args...)
        EnzymeCore.autodiff_deferred(Forward, Const(f), Const, Const(ctx), args...)
        return nothing
    end

    function aug_fwd(ctx, f::FT, ::Val{ModifiedBetween}, subtape, args...) where {ModifiedBetween, FT}
        forward, reverse = EnzymeCore.autodiff_deferred_thunk(ReverseSplitModified(ReverseSplitWithPrimal, Val(ModifiedBetween)), Const{Core.Typeof(f)}, Const, Const{Core.Typeof(ctx)}, map(Core.Typeof, args)...)
        subtape[__groupindex(ctx)] = forward(Const(f), Const(ctx), args...)[1]
        return nothing
    end

    function rev(ctx, f::FT, ::Val{ModifiedBetween}, subtape, args...) where {ModifiedBetween, FT}
        forward, reverse = EnzymeCore.autodiff_deferred_thunk(ReverseSplitModified(ReverseSplitWithPrimal, Val(ModifiedBetween)), Const{Core.Typeof(f)}, Const, Const{Core.Typeof(ctx)}, map(Core.Typeof, args)...)
        tp = subtape[__groupindex(ctx)]
        reverse(Const(f), Const(ctx), args..., tp)
        return nothing
    end

    function EnzymeRules.forward(func::Const{<:Kernel}, ::Type{Const{Nothing}}, args...; ndrange=nothing, workgroupsize=nothing)
        kernel = func.val
        f = kernel.f
        fwd_kernel = similar(kernel, fwd)

        fwd_kernel(f, args...; ndrange, workgroupsize)
    end

    function EnzymeRules.augmented_primal(config::Config, func::Const{<:Kernel{CPU}}, ::Type{Const{Nothing}}, args::Vararg{Any, N}; ndrange=nothing, workgroupsize=nothing) where N
        kernel = func.val
        f = kernel.f

        ndrange, workgroupsize, iterspace, dynamic = launch_config(kernel, ndrange, workgroupsize)
        block = first(blocks(iterspace))

        ctx = mkcontext(kernel, block, ndrange, iterspace, dynamic)
        ctxTy = Core.Typeof(ctx) # CompilerMetadata{ndrange(kernel), Core.Typeof(dynamic)}

        # TODO autodiff_deferred on the func.val
        ModifiedBetween = Val((overwritten(config)[1], false, overwritten(config)[2:end]...))

        FT = Const{Core.Typeof(f)}

        arg_refs = ntuple(Val(N)) do i
            Base.@_inline_meta
            if args[i] isa Active
                Ref(EnzymeCore.make_zero(args[i]))
            else
                nothing
            end
        end
        args2 = ntuple(Val(N)) do i
            Base.@_inline_meta
            if args[i] isa Active
                Duplicated(Ref(args[i]), arg_refs[i])
            else
                args[i]
            end
        end

        # TODO in KA backends like CUDAKernels, etc have a version with a parent job type
        TapeType = EnzymeCore.tape_type(ReverseSplitModified(ReverseSplitWithPrimal, ModifiedBetween), FT, Const,  Const{ctxTy}, map(Core.Typeof, args2)...)


        subtape = Array{TapeType}(undef, size(blocks(iterspace)))

        aug_kernel = similar(kernel, aug_fwd)

        aug_kernel(f, ModifiedBetween, subtape, args2...; ndrange, workgroupsize)

        # TODO the fact that ctxTy is type unstable means this is all type unstable.
        # Since custom rules require a fixed return type, explicitly cast to Any, rather
        # than returning a AugmentedReturn{Nothing, Nothing, T} where T.

        res =  AugmentedReturn{Nothing, Nothing, Tuple{Array, typeo(arg_refs)}}(nothing, nothing, (subtape, arg_refs))
        return res
    end

    function EnzymeRules.reverse(config::Config, func::Const{<:Kernel}, ::Type{<:EnzymeCore.Annotation}, tape, args::Vararg{Any, N}; ndrange=nothing, workgroupsize=nothing) where N
        sub_tape, arg_refs = tape

        args2 = ntuple(Val(N)) do i
            Base.@_inline_meta
            if args[i] isa Active
                Duplicated(Ref(args[i]), arg_refs[i])
            else
                args[i]
            end
        end

        kernel = func.val
        f = kernel.f

        ModifiedBetween = Val((overwritten(config)[1], false, overwritten(config)[2:end]...))

        rev_kernel = similar(func.val, rev)
        rev_kernel(f, ModifiedBetween, subtape, args2...; ndrange, workgroupsize)
        return ntuple(Val(N)) do i
            Base.@_inline_meta
            if args[i] isa Active
                arg_refs[i][]
            else
                nothing
            end
        end
    end
end
