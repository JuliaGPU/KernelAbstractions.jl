# Split reverse mode of the kernel body, with the settings of `config` (runtime activity, strong zero).
split_mode(config, ModifiedBetween) =
    EnzymeRules.set_config(ReverseSplitModified(ReverseSplitWithPrimal, ModifiedBetween), config)

_unwrap_const_type(arg) = arg
_unwrap_const_type(arg::Const{<:Type}) = arg.val
_rewrap_const_type(arg) = arg
_rewrap_const_type(::Type{T}) where {T} = Const{Type{T}}(T)

function fwd(ctx, config, f, args...)
    # no argument is `@Const`: the kernel's own arguments are wrapped in annotations
    ctx isa ConstArgs && return nothing
    args = map(_rewrap_const_type, args)
    EnzymeCore.autodiff_deferred(EnzymeRules.set_config(Forward, config), Const(f), Const{Nothing}, Const(ctx), args...)
    return nothing
end

function EnzymeRules.forward(
        config,
        func::Const{<:Kernel},
        ::Type{Const{Nothing}},
        args...;
        ndrange = nothing,
        workgroupsize = nothing,
    )
    kernel = func.val
    f = kernel.f
    fwd_kernel = similar(kernel, fwd)

    # `Const{Type{T}}` is not a bitstype, so it cannot be passed to kernels.
    # Pass the type itself, and re-wrap it as `Const` in the kernel (`fwd`).
    args = map(_unwrap_const_type, args)
    return fwd_kernel(config, f, args...; ndrange, workgroupsize)
end

# The context that launching `aug_fwd` or `rev` on the backend of `kernel` creates, as the
# kernel receives it (see `launch_tuple`). The tape type depends on its type.
function _enzyme_mkcontext(kernel::Kernel, ndrange, workgroupsize, iterspace)
    launch = select_launch(kernel, workgroupsize, iterspace)
    return argconvert(kernel, mkcontext(kernel, ndrange, iterspace, launch))
end

# A launch whose workgroups cover the ndrange gets a context without the bounds check, which
# only differs in its type. The kernel body is differentiated with the checked one, so that
# the tape type doesn't depend on the launch.
@inline checked_context(ctx::CompilerMetadata{NDRange, DynamicCheck}) where {NDRange} = ctx
@inline checked_context(ctx::CompilerMetadata{NDRange}) where {NDRange} =
    CompilerMetadata{NDRange, DynamicCheck}(ctx.ndrange, ctx.iterspace; ctx.launch, ctx.subgroups)

_augmented_return(::Kernel, subtape, arg_refs, tape_type, heap) =
    AugmentedReturn{Nothing, Nothing, Any}(nothing, nothing, (subtape, arg_refs, tape_type, heap))

function _create_tape_kernel(
        kernel::Kernel,
        Mode,
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
        Mode,
        FT,
        Const{Nothing},
        Const{ctxTy},
        dev_TT...,
    )

    # Allocate per thread
    nthreads = _ka_nthreads(ndrange, iterspace)
    subtape = allocate(backend(kernel), TapeType, nthreads)
    heap = _tape_heap(backend(kernel), TapeType, nthreads)

    aug_kernel = similar(kernel, aug_fwd)
    return TapeType, subtape, heap, aug_kernel
end

function aug_fwd(
        ctx,
        f::FT,
        mode::Mode,
        subtape,
        ::Val{TapeType},
        args...,
    ) where {Mode, FT, TapeType}
    ctx isa ConstArgs && return nothing
    checked = checked_context(ctx)
    # A2 = Const{Nothing} -- since f->Nothing
    forward, _ = EnzymeCore.autodiff_deferred_thunk(
        mode,
        TapeType,
        Const{Core.Typeof(f)},
        Const{Nothing},
        Const{Core.Typeof(checked)},
        map(Core.Typeof, args)...,
    )

    # On the GPU: F is a per thread function
    # On the GPU: subtape::Vector
    if __validindex(ctx)
        I = __index_Global_Linear(ctx)
        subtape[I] = forward(Const(f), Const(checked), args...)[1]
    end
    return nothing
end

function rev(
        ctx,
        f::FT,
        mode::Mode,
        subtape,
        ::Val{TapeType},
        args...,
    ) where {Mode, FT, TapeType}
    ctx isa ConstArgs && return nothing
    checked = checked_context(ctx)
    # XXX: TapeType and A2 as args to autodiff_deferred_thunk
    _, reverse = EnzymeCore.autodiff_deferred_thunk(
        mode,
        TapeType,
        Const{Core.Typeof(f)},
        Const{Nothing},
        Const{Core.Typeof(checked)},
        map(Core.Typeof, args)...,
    )
    if __validindex(ctx)
        I = __index_Global_Linear(ctx)
        tp = subtape[I]
        reverse(Const(f), Const(checked), args..., tp)
    end
    return nothing
end

function EnzymeRules.augmented_primal(
        config::RevConfig,
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
    ctx = _enzyme_mkcontext(kernel, ndrange, workgroupsize, iterspace)
    ctxTy = Core.Typeof(ctx)
    # TODO autodiff_deferred on the func.val
    ModifiedBetween = Val((overwritten(config)[1], false, overwritten(config)[2:end]...))

    FT = Const{Core.Typeof(f)}

    arg_refs = ntuple(Val(N)) do i
        Base.@_inline_meta
        if args[i] isa Active
            error(
                """
                Active kernel arguments not supported, got
                \n$(typeof(args[i]))
                \nFor argument $i of kernel
                \n$(func.val).
                """
            )
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
    Mode = split_mode(config, ModifiedBetween)
    TapeType, subtape, heap, aug_kernel = _create_tape_kernel(
        kernel,
        Mode,
        FT,
        ctxTy,
        ndrange,
        iterspace,
        args2...,
    )
    _with_tape_heap(heap) do
        aug_kernel(f, Mode, subtape, Val(TapeType), args2...; ndrange, workgroupsize)
    end

    # TODO the fact that ctxTy is type unstable means this is all type unstable.
    # Since custom rules require a fixed return type, explicitly cast to Any, rather
    # than returning a AugmentedReturn{Nothing, Nothing, T} where T.
    return _augmented_return(kernel, subtape, arg_refs, TapeType, heap)
end

function EnzymeRules.reverse(
        config::RevConfig,
        func::Const{<:Kernel},
        ::Type{<:EnzymeCore.Annotation},
        tape,
        args::Vararg{Any, N};
        ndrange = nothing,
        workgroupsize = nothing,
    ) where {N}
    subtape, arg_refs, tape_type, heap = tape

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
    Mode = split_mode(config, ModifiedBetween)
    rev_kernel = similar(kernel, rev)
    _with_tape_heap(heap) do
        rev_kernel(
            f,
            Mode,
            subtape,
            Val(tape_type),
            args2...;
            ndrange,
            workgroupsize,
        )
    end
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
