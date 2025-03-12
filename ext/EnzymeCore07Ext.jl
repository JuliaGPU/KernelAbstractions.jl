function fwd(ctx, f, args...)
    EnzymeCore.autodiff_deferred(Forward, Const(f), Const{Nothing}, Const(ctx), args...)
    return nothing
end

function EnzymeRules.forward(
        func::Const{<:Kernel},
        ::Type{Const{Nothing}},
        args...;
        ndrange = nothing,
        workgroupsize = nothing,
    )
    kernel = func.val
    f = kernel.f
    fwd_kernel = similar(kernel, fwd)

    return fwd_kernel(f, args...; ndrange, workgroupsize)
end

_enzyme_mkcontext(kernel::Kernel, ndrange, iterspace, dynamic) =
    mkcontext(kernel, ndrange, iterspace)

_augmented_return(::Kernel, subtape, arg_refs, tape_type) =
    AugmentedReturn{Nothing, Nothing, Any}(nothing, nothing, (subtape, arg_refs, tape_type))

function _create_tape_kernel(
        kernel::Kernel,
        ModifiedBetween,
        FT,
        ctxTy,
        ndrange,
        iterspace,
        args2...,
    )
    # For peeking at the TapeType we need to first construct a correct compilation job
    # this requires the use of the device side representation of arguments.
    # So we convert the arguments here, this is a bit wasteful since the `aug_kernel` call
    # will later do the same.
    dev_args2 = ((argconvert(kernel, a) for a in args2)...,)
    dev_TT = map(Core.Typeof, dev_args2)

    job =
        EnzymeCore.compiler_job_from_backend(backend(kernel), typeof(() -> return), Tuple{})
    TapeType = EnzymeCore.tape_type(
        job,
        ReverseSplitModified(ReverseSplitWithPrimal, ModifiedBetween),
        FT,
        Const{Nothing},
        Const{ctxTy},
        dev_TT...,
    )

    # Allocate per thread
    subtape = allocate(backend(kernel), TapeType, prod(ndrange))

    aug_kernel = similar(kernel, aug_fwd)
    return TapeType, subtape, aug_kernel
end

function aug_fwd(
        ctx,
        f::FT,
        ::Val{ModifiedBetween},
        subtape,
        ::Val{TapeType},
        args...,
    ) where {ModifiedBetween, FT, TapeType}
    # A2 = Const{Nothing} -- since f->Nothing
    forward, _ = EnzymeCore.autodiff_deferred_thunk(
        ReverseSplitModified(ReverseSplitWithPrimal, Val(ModifiedBetween)),
        TapeType,
        Const{Core.Typeof(f)},
        Const{Nothing},
        Const{Core.Typeof(ctx)},
        map(Core.Typeof, args)...,
    )

    # On the GPU: F is a per thread function
    # On the GPU: subtape::Vector
    if __validindex(ctx)
        I = __index_Global_Linear(ctx)
        subtape[I] = forward(Const(f), Const(ctx), args...)[1]
    end
    return nothing
end

function rev(
        ctx,
        f::FT,
        ::Val{ModifiedBetween},
        subtape,
        ::Val{TapeType},
        args...,
    ) where {ModifiedBetween, FT, TapeType}
    # XXX: TapeType and A2 as args to autodiff_deferred_thunk
    _, reverse = EnzymeCore.autodiff_deferred_thunk(
        ReverseSplitModified(ReverseSplitWithPrimal, Val(ModifiedBetween)),
        TapeType,
        Const{Core.Typeof(f)},
        Const{Nothing},
        Const{Core.Typeof(ctx)},
        map(Core.Typeof, args)...,
    )
    if __validindex(ctx)
        I = __index_Global_Linear(ctx)
        tp = subtape[I]
        reverse(Const(f), Const(ctx), args..., tp)
    end
    return nothing
end

function EnzymeRules.augmented_primal(
        config::Config,
        func::Const{<:Kernel},
        ::Type{Const{Nothing}},
        args::Vararg{Any, N};
        ndrange = nothing,
        workgroupsize = nothing,
    ) where {N}
    kernel = func.val
    f = kernel.f

    ndrange, workgroupsize, iterspace, dynamic =
        launch_config(kernel, ndrange, workgroupsize)
    ctx = _enzyme_mkcontext(kernel, ndrange, iterspace, dynamic)
    ctxTy = Core.Typeof(ctx) # CompilerMetadata{ndrange(kernel), Core.Typeof(dynamic)}
    # TODO autodiff_deferred on the func.val
    ModifiedBetween = Val((overwritten(config)[1], false, overwritten(config)[2:end]...))

    FT = Const{Core.Typeof(f)}

    arg_refs = ntuple(Val(N)) do i
        Base.@_inline_meta
        if args[i] isa Active
            error("Active kernel arguments not supported")
        else
            nothing
        end
    end
    args2 = ntuple(Val(N)) do i
        Base.@_inline_meta
        if args[i] isa Active
            MixedDuplicated(args[i].val, arg_refs[i])
        else
            args[i]
        end
    end

    TapeType, subtape, aug_kernel = _create_tape_kernel(
        kernel,
        ModifiedBetween,
        FT,
        ctxTy,
        ndrange,
        iterspace,
        args2...,
    )
    aug_kernel(f, ModifiedBetween, subtape, Val(TapeType), args2...; ndrange, workgroupsize)

    # TODO the fact that ctxTy is type unstable means this is all type unstable.
    # Since custom rules require a fixed return type, explicitly cast to Any, rather
    # than returning a AugmentedReturn{Nothing, Nothing, T} where T.
    return _augmented_return(kernel, subtape, arg_refs, TapeType)
end

function EnzymeRules.reverse(
        config::Config,
        func::Const{<:Kernel},
        ::Type{<:EnzymeCore.Annotation},
        tape,
        args::Vararg{Any, N};
        ndrange = nothing,
        workgroupsize = nothing,
    ) where {N}
    subtape, arg_refs, tape_type = tape

    args2 = ntuple(Val(N)) do i
        Base.@_inline_meta
        if args[i] isa Active
            MixedDuplicated(args[i].val, arg_refs[i])
        else
            args[i]
        end
    end

    kernel = func.val
    f = kernel.f

    ModifiedBetween = Val((overwritten(config)[1], false, overwritten(config)[2:end]...))

    rev_kernel = similar(kernel, rev)
    rev_kernel(
        f,
        ModifiedBetween,
        subtape,
        Val(tape_type),
        args2...;
        ndrange,
        workgroupsize,
    )
    res = ntuple(Val(N)) do i
        Base.@_inline_meta
        if args[i] isa Active
            arg_refs[i][]
        else
            nothing
        end
    end
    # Reverse synchronization right after the kernel launch
    synchronize(backend(kernel))
    return res
end

# Synchronize rules
# TODO: Right now we do the synchronization as part of the kernel launch in the augmented primal
#       and reverse rules. This is not ideal, as we would want to launch the kernel in the reverse
#       synchronize rule and then synchronize where the launch was. However, with the current
#       kernel semantics this ensures correctness for now.
function EnzymeRules.augmented_primal(
        config::Config,
        func::Const{typeof(synchronize)},
        ::Type{Const{Nothing}},
        backend::T,
    ) where {T <: EnzymeCore.Annotation}
    synchronize(backend.val)
    return AugmentedReturn(nothing, nothing, nothing)
end

function EnzymeRules.reverse(
        config::Config,
        func::Const{typeof(synchronize)},
        ::Type{Const{Nothing}},
        tape,
        backend,
    )
    # noop for now
    return (nothing,)
end
