# https://github.com/EnzymeAD/Enzyme.jl/issues/1516
# On the CPU `autodiff_deferred` can deadlock.
# Hence a specialized CPU version
function cpu_fwd(ctx, config, f, args...)
    EnzymeCore.autodiff(EnzymeCore.set_runtime_activity(Forward, config), Const(f), Const{Nothing}, Const(ctx), args...)
    return nothing
end

function gpu_fwd(ctx, config, f, args...)
    EnzymeCore.autodiff_deferred(EnzymeCore.set_runtime_activity(Forward, config), Const(f), Const{Nothing}, Const(ctx), args...)
    return nothing
end

function EnzymeRules.forward(
        config,
        func::Const{<:Kernel{CPU}},
        ::Type{Const{Nothing}},
        args...;
        ndrange = nothing,
        workgroupsize = nothing,
    )
    kernel = func.val
    f = kernel.f
    fwd_kernel = similar(kernel, cpu_fwd)

    return fwd_kernel(config, f, args...; ndrange, workgroupsize)
end

function EnzymeRules.forward(
        config,
        func::Const{<:Kernel{<:GPU}},
        ::Type{Const{Nothing}},
        args...;
        ndrange = nothing,
        workgroupsize = nothing,
    )
    kernel = func.val
    f = kernel.f
    fwd_kernel = similar(kernel, gpu_fwd)

    return fwd_kernel(config, f, args...; ndrange, workgroupsize)
end

_enzyme_mkcontext(kernel::Kernel{CPU}, ndrange, iterspace, dynamic) =
    mkcontext(kernel, first(blocks(iterspace)), ndrange, iterspace, dynamic)
_enzyme_mkcontext(kernel::Kernel{<:GPU}, ndrange, iterspace, dynamic) =
    mkcontext(kernel, ndrange, iterspace)

_augmented_return(::Kernel{CPU}, subtape, arg_refs, tape_type) =
    AugmentedReturn{Nothing, Nothing, Tuple{Array, typeof(arg_refs), typeof(tape_type)}}(
    nothing,
    nothing,
    (subtape, arg_refs, tape_type),
)
_augmented_return(::Kernel{<:GPU}, subtape, arg_refs, tape_type) =
    AugmentedReturn{Nothing, Nothing, Any}(nothing, nothing, (subtape, arg_refs, tape_type))

function _create_tape_kernel(
        kernel::Kernel{CPU},
        Mode,
        FT,
        ctxTy,
        ndrange,
        iterspace,
        args2...,
    )
    TapeType = EnzymeCore.tape_type(
        Mode,
        FT,
        Const{Nothing},
        Const{ctxTy},
        map(Core.Typeof, args2)...,
    )
    subtape = Array{TapeType}(undef, size(blocks(iterspace)))
    aug_kernel = similar(kernel, cpu_aug_fwd)
    return TapeType, subtape, aug_kernel
end

function _create_tape_kernel(
        kernel::Kernel{<:GPU},
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
    dev_TT = map(a -> _device_argtype(Core.Typeof(a)), dev_args2)

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

    aug_kernel = similar(kernel, gpu_aug_fwd)
    return TapeType, subtape, aug_kernel
end

_create_rev_kernel(kernel::Kernel{CPU}) = similar(kernel, cpu_rev)
_create_rev_kernel(kernel::Kernel{<:GPU}) = similar(kernel, gpu_rev)

function cpu_aug_fwd(
        ctx,
        f::FT,
        mode::Mode,
        subtape,
        ::Val{TapeType},
        args...,
    ) where {Mode, FT, TapeType}
    # A2 = Const{Nothing} -- since f->Nothing
    forward, _ = EnzymeCore.autodiff_thunk(
        mode,
        Const{Core.Typeof(f)},
        Const{Nothing},
        Const{Core.Typeof(ctx)},
        map(Core.Typeof, args)...,
    )

    # On the CPU: F is a per block function
    # On the CPU: subtape::Vector{Vector}
    I = __index_Group_Cartesian(ctx, CartesianIndex(1, 1)) #=fake=#
    subtape[I] = forward(Const(f), Const(ctx), args...)[1]
    return nothing
end

function cpu_rev(
        ctx,
        f::FT,
        mode::Mode,
        subtape,
        ::Val{TapeType},
        args...,
    ) where {Mode, FT, TapeType}
    _, reverse = EnzymeCore.autodiff_thunk(
        mode,
        Const{Core.Typeof(f)},
        Const{Nothing},
        Const{Core.Typeof(ctx)},
        map(Core.Typeof, args)...,
    )
    I = __index_Group_Cartesian(ctx, CartesianIndex(1, 1)) #=fake=#
    tp = subtape[I]
    reverse(Const(f), Const(ctx), args..., tp)
    return nothing
end

# GPU support
function gpu_aug_fwd(
        ctx,
        f::FT,
        mode::Mode,
        subtape,
        ::Val{TapeType},
        args...,
    ) where {Mode, FT, TapeType}
    args = _device_args(args)
    # A2 = Const{Nothing} -- since f->Nothing
    forward, _ = EnzymeCore.autodiff_deferred_thunk(
        mode,
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

function gpu_rev(
        ctx,
        f::FT,
        mode::Mode,
        subtape,
        ::Val{TapeType},
        args...,
    ) where {Mode, FT, TapeType}
    args = _device_args(args)
    # XXX: TapeType and A2 as args to autodiff_deferred_thunk
    _, reverse = EnzymeCore.autodiff_deferred_thunk(
        mode,
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

# Active arguments
# On the CPU the kernel accumulates the adjoint of an `Active` argument into a host `Ref`,
# passed as `MixedDuplicated`. With a batch width `W > 1` there is one `Ref` per lane,
# passed as `BatchMixedDuplicated`. On the GPU the kernel cannot write to a host `Ref`.
# Instead the shadow lives in a `W`-element device array, one element per lane. The primal
# is passed by value. The host passes both as a `DeviceMixed`, and the kernel turns it into
# `MixedDuplicated(val, pointer(dval))` or
# `BatchMixedDuplicated(val, ntuple(k -> pointer(dval, k), W))`. Enzyme uses atomic adds
# for shadow updates on GPU targets, so all threads can accumulate into the same element.
_active_ref(::Kernel{CPU}, val, ::Val{1}) = Ref(EnzymeCore.make_zero(val))
_active_ref(::Kernel{CPU}, val, ::Val{W}) where {W} =
    ntuple(_ -> Ref(EnzymeCore.make_zero(val)), Val(W))
function _active_ref(kernel::Kernel{<:GPU}, val::T, ::Val{W}) where {T, W}
    dbox = allocate(backend(kernel), T, W)
    copyto!(dbox, fill(EnzymeCore.make_zero(val), W))
    return dbox
end

struct DeviceMixed{W, T, D}
    val::T
    dval::D
end
DeviceMixed{W}(val::T, dval::D) where {W, T, D} = DeviceMixed{W, T, D}(val, dval)
Adapt.adapt_structure(to, x::DeviceMixed{W}) where {W} =
    DeviceMixed{W}(x.val, Adapt.adapt(to, x.dval))

_active_arg(arg, ref::Base.RefValue, ::Val) = MixedDuplicated(arg.val, ref)
_active_arg(arg, refs::NTuple{W, Base.RefValue}, ::Val) where {W} =
    BatchMixedDuplicated(arg.val, refs)
_active_arg(arg, ref::AbstractArray, ::Val{W}) where {W} = DeviceMixed{W}(arg.val, ref)
_active_arg(arg, ::Nothing, ::Val) = arg

_active_args(args::NTuple{N, Any}, arg_refs, width) where {N} = ntuple(Val(N)) do i
    Base.@_inline_meta
    _active_arg(args[i], arg_refs[i], width)
end

_active_refs(kernel, args::NTuple{N, Any}, width) where {N} = ntuple(Val(N)) do i
    Base.@_inline_meta
    args[i] isa Active ? _active_ref(kernel, args[i].val, width) : nothing
end

_active_result(ref::Base.RefValue, ::Val{1}) = ref[]
_active_result(refs::NTuple{W, Base.RefValue}, ::Val{W}) where {W} = map(getindex, refs)
_active_result(ref::AbstractArray, ::Val{1}) = only(Array(ref))
_active_result(ref::AbstractArray, ::Val{W}) where {W} = NTuple{W}(Array(ref))

_active_restype(::Type{T}, ::Val{1}) where {T} = T
_active_restype(::Type{T}, ::Val{W}) where {T, W} = NTuple{W, T}

# On the GPU the tape, and hence `arg_refs`, is not inferred. Dispatch on the argument
# annotation, whose type is known, to assert the result type of each position.
_active_result_for(::Active{T}, ref, width) where {T} =
    _active_result(ref, width)::_active_restype(T, width)
_active_result_for(arg, ref, width) = nothing
_active_results(args::NTuple{N, Any}, arg_refs, width) where {N} =
    map((arg, i) -> _active_result_for(arg, arg_refs[i], width), args, ntuple(identity, Val(N)))

# Device side: build the `(Batch)MixedDuplicated` from the device shadow.
@inline _device_arg(arg) = arg
@inline _device_arg(arg::DeviceMixed{1}) = MixedDuplicated(arg.val, pointer(arg.dval))
@inline _device_arg(arg::DeviceMixed{W}) where {W} =
    BatchMixedDuplicated(arg.val, ntuple(k -> pointer(arg.dval, k), Val(W)))
@inline _device_args(args::NTuple{N, Any}) where {N} = ntuple(Val(N)) do i
    Base.@_inline_meta
    _device_arg(args[i])
end
_device_argtype(::Type{T}) where {T} = T
_device_argtype(::Type{DeviceMixed{1, T, D}}) where {T, D} = MixedDuplicated{T}
_device_argtype(::Type{DeviceMixed{W, T, D}}) where {W, T, D} = BatchMixedDuplicated{T, W}

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
    ctx = _enzyme_mkcontext(kernel, ndrange, iterspace, dynamic)
    ctxTy = Core.Typeof(ctx) # CompilerMetadata{ndrange(kernel), Core.Typeof(dynamic)}
    # TODO autodiff_deferred on the func.val
    ModifiedBetween = Val((overwritten(config)[1], false, overwritten(config)[2:end]...))

    width = Val(EnzymeRules.width(config))
    arg_refs = _active_refs(kernel, args, width)
    args2 = _active_args(args, arg_refs, width)
    FT = Const{Core.Typeof(f)}
    Mode = EnzymeCore.set_runtime_activity(ReverseSplitModified(ReverseSplitWithPrimal, ModifiedBetween), config)
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
    Mode = EnzymeCore.set_runtime_activity(ReverseSplitModified(ReverseSplitWithPrimal, ModifiedBetween), config)
    rev_kernel = _create_rev_kernel(kernel)
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
