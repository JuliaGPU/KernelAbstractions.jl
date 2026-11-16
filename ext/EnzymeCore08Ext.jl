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

_augmented_return(::Kernel, subtape, arg_refs, tape_type) =
    AugmentedReturn{Nothing, Nothing, Any}(nothing, nothing, (subtape, arg_refs, tape_type))

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
    dev_TT = map(a -> Core.Typeof(_device_arg(a)), dev_args2)

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
    subtape = allocate(backend(kernel), TapeType, prod(ndrange))

    aug_kernel = similar(kernel, aug_fwd)
    return TapeType, subtape, aug_kernel
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
    args = _device_args(args)
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
    args = _device_args(args)
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

# Active arguments
# A kernel cannot write to a host `Ref`, so it cannot accumulate the adjoint of an `Active`
# argument into the shadow of a `MixedDuplicated`. Instead the shadow lives in a device
# array with one element per lane of the batch, and the primal is passed by value. The host
# passes both as a `DeviceMixed`, and the kernel turns it into
# `MixedDuplicatedPtr(val, pointer(dval))` or
# `BatchMixedDuplicatedPtr(val, ntuple(k -> pointer(dval, k), W))`. Enzyme updates shadows
# atomically in device code, so all work-items can accumulate into the same element.
struct DeviceMixed{W, T, D}
    val::T
    dval::D
end
DeviceMixed{W}(val::T, dval::D) where {W, T, D} = DeviceMixed{W, T, D}(val, dval)
Adapt.adapt_structure(to, x::DeviceMixed{W}) where {W} =
    DeviceMixed{W}(x.val, Adapt.adapt(to, x.dval))

# The shadow of the `Active` argument `i` of `kernel`
_active_ref(kernel, i, arg, width) = nothing
function _active_ref(kernel::Kernel, i, arg::Active{T}, ::Val{W}) where {T, W}
    if !isdefined(EnzymeCore, :MixedDuplicatedPtr) || !isbitstype(T)
        error(
            """
            Active kernel arguments need an isbits type and EnzymeCore with `MixedDuplicatedPtr`, got
            \n$(typeof(arg))
            \nFor argument $i of kernel
            \n$(kernel).
            """
        )
    end
    dval = allocate(backend(kernel), T, W)
    copyto!(dval, fill(EnzymeCore.make_zero(arg.val), W))
    return dval
end

_active_refs(kernel, args::NTuple{N, Any}, width) where {N} = ntuple(Val(N)) do i
    Base.@_inline_meta
    _active_ref(kernel, i, args[i], width)
end

_active_arg(arg, ::Nothing, ::Val) = arg
_active_arg(arg, dval, ::Val{W}) where {W} = DeviceMixed{W}(arg.val, dval)

_active_args(args::NTuple{N, Any}, arg_refs, width) where {N} = ntuple(Val(N)) do i
    Base.@_inline_meta
    _active_arg(args[i], arg_refs[i], width)
end

# The tape, and hence `arg_refs`, is not inferred. Dispatch on the argument annotation,
# whose type is known, to assert the result type of each position.
_active_result(arg, dval, width) = nothing
_active_result(::Active{T}, dval, ::Val{1}) where {T} = only(Array(dval))::T
_active_result(::Active{T}, dval, ::Val{W}) where {T, W} = NTuple{W, T}(Array(dval))::NTuple{W, T}
_active_results(args::NTuple{N, Any}, arg_refs, width) where {N} =
    map((arg, i) -> _active_result(arg, arg_refs[i], width), args, ntuple(identity, Val(N)))

# In the kernel: the annotation with the shadow of a `DeviceMixed`. Also called on the host,
# with the converted arguments, for the argument types of the kernel body.
@inline _device_arg(arg) = arg
@inline _device_arg(arg::DeviceMixed{1}) =
    EnzymeCore.MixedDuplicatedPtr(arg.val, pointer(arg.dval))
@inline _device_arg(arg::DeviceMixed{W}) where {W} =
    EnzymeCore.BatchMixedDuplicatedPtr(arg.val, ntuple(k -> pointer(arg.dval, k), Val(W)))
@inline _device_args(args::NTuple{N, Any}) where {N} = ntuple(Val(N)) do i
    Base.@_inline_meta
    _device_arg(args[i])
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

    width = Val(EnzymeRules.width(config))
    arg_refs = _active_refs(kernel, args, width)
    args2 = _active_args(args, arg_refs, width)
    Mode = split_mode(config, ModifiedBetween)
    TapeType, subtape, aug_kernel = _create_tape_kernel(
        kernel,
        Mode,
        FT,
        ctxTy,
        ndrange,
        iterspace,
        args2...,
    )
    aug_kernel(f, Mode, subtape, Val(TapeType), args2...; ndrange, workgroupsize)

    # TODO the fact that ctxTy is type unstable means this is all type unstable.
    # Since custom rules require a fixed return type, explicitly cast to Any, rather
    # than returning a AugmentedReturn{Nothing, Nothing, T} where T.
    return _augmented_return(kernel, subtape, arg_refs, TapeType)
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
    subtape, arg_refs, tape_type = tape

    kernel = func.val
    width = Val(EnzymeRules.width(config))
    args2 = _active_args(args, arg_refs, width)
    f = kernel.f

    ModifiedBetween = Val((overwritten(config)[1], false, overwritten(config)[2:end]...))
    Mode = split_mode(config, ModifiedBetween)
    rev_kernel = similar(kernel, rev)
    rev_kernel(
        f,
        Mode,
        subtape,
        Val(tape_type),
        args2...;
        ndrange,
        workgroupsize,
    )
    # Reverse synchronization right after the kernel launch
    synchronize(backend(kernel))
    return _active_results(args, arg_refs, width)
end
