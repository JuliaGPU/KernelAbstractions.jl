###
# Launching `@kernel` kernels on a KernelInterface backend
#
# Every backend implementing KernelInterface launches `@kernel` kernels with the methods
# below. Backends customize them through the hooks documented in `implementations.md`
# (`compiler_options`, and KernelInterface's `launch_configuration`) instead of
# reimplementing the launch.
###

"""
    mkcontext(kernel::Kernel, ndrange, iterspace, [launch])

The hidden context argument for launching `kernel` over `ndrange`, partitioned as
`iterspace`, with the launch configuration `launch` (see [`select_launch`](@ref)).
"""
mkcontext(kernel::Kernel, _ndrange, iterspace) =
    CompilerMetadata{ndrange(kernel), DynamicCheck}(_ndrange, iterspace)
mkcontext(kernel::Kernel, _ndrange, iterspace, launch) =
    CompilerMetadata{ndrange(kernel), DynamicCheck}(_ndrange, iterspace; launch)
mkcontext(kernel::Kernel, I, _ndrange, iterspace, ::Dynamic) where {Dynamic} =
    CompilerMetadata{ndrange(kernel), Dynamic}(I, _ndrange, iterspace)

"""
    launch_config(kernel::Kernel, ndrange, workgroupsize)

Normalize the launch arguments of `kernel`, and partition the `ndrange`. Returns the
`ndrange` (`nothing` if it's static), the `workgroupsize` (`nothing` if it will be tuned),
the iteration space and whether it needs bounds checks. If the workgroup size will be
tuned, the iteration space is preliminary: it uses the extents of the `ndrange` as the
workgroup size.
"""
function launch_config(kernel::Kernel, _ndrange, _workgroupsize)
    if _ndrange isa Integer
        _ndrange = (_ndrange,)
    end
    if _workgroupsize isa Integer
        _workgroupsize = (_workgroupsize,)
    end

    iterspace, dynamic = if workgroupsize(kernel) <: DynamicSize && _workgroupsize === nothing
        # use the extents of the ndrange as preliminary workgroupsize for autotuning
        partition(kernel, _ndrange, extents(something(_ndrange, static_ndrange(kernel))))
    else
        # this also checks that a given ndrange agrees with a static one
        partition(kernel, _ndrange, _workgroupsize)
    end
    if ndrange(kernel) <: StaticSize
        _ndrange = nothing
    end

    return _ndrange, _workgroupsize, iterspace, dynamic
end

"""
    compiler_options(kernel::Kernel)::NamedTuple

Backend-specific compiler options for compiling `kernel` with
[`KI.kernel_function`](@ref KernelInterface.kernel_function), e.g. a hint derived from its
static workgroup size (CUDA.jl passes `maxthreads`). Backends **may** implement this for
their backend type; the default is no options.
"""
compiler_options(::Kernel) = (;)

static_ndrange(kernel::Kernel) = ndrange(kernel) <: StaticSize ? get(ndrange(kernel)) : nothing

# the product of `dims`, saturated at `typemax(Int)`
function saturated_prod(dims::Dims)
    n = 1
    for d in dims
        n, overflow = Base.mul_with_overflow(n, d)
        overflow && return typemax(Int)
    end
    return n
end

argconvert(kernel::Kernel{<:KI.Backend}, arg) = KI.argconvert(backend(kernel), arg)

# The arguments are passed on as a tuple: Julia doesn't turn a splat of more than 32
# elements into a direct call, and a method with both varargs and keyword arguments splats
# them into its body. So the keyword method is defined explicitly, as Base does for
# `invokelatest`.
(obj::Kernel{<:KI.Backend})(args::Vararg{Any, N}) where {N} = launch_tuple(obj, args)
Core.kwcall(kwargs::NamedTuple, obj::Kernel{<:KI.Backend}, args::Vararg{Any, N}) where {N} =
    launch_tuple(obj, args; kwargs...)

function launch_tuple(obj::Kernel, args::Tuple; ndrange = nothing, workgroupsize = nothing, gridstride = false)
    ndrange, workgroupsize, iterspace, dynamic = launch_config(obj, ndrange, workgroupsize)
    # nothing to launch (or compile) for an empty ndrange
    any(iszero, size(blocks(iterspace))) && return nothing

    if gridstride === true && !occupancy_gridstride(backend(obj))
        gridstride = false
    end
    if gridstride !== false
        maxgroups = check_gridstride(gridstride)
        Base.inferencebarrier(launch_gridstride)(obj, maxgroups, ndrange, workgroupsize, iterspace, args)
        return nothing
    end

    # launch on an N-d grid, computing indices in a narrow type, if possible. this doesn't
    # depend on the tuned workgroup size, so the context (and thus the kernel) doesn't either.
    launch = select_launch(obj, workgroupsize, iterspace)
    T = narrow_index_type(backend(obj))
    if launch === NDLaunch{T}()
        # the common case, specialized statically
        launch_kernel(obj, NDLaunch{T}(), ndrange, workgroupsize, iterspace, args)
    elseif !(launch isa NDLaunch) && exceeds_grid(obj, workgroupsize, iterspace)
        # the grid can't hold a group per group of the iteration space: launch as many
        # groups as it holds, each looping over several groups of the iteration space
        Base.inferencebarrier(launch_gridstride)(obj, typemax(Int), ndrange, workgroupsize, iterspace, args)
    else
        # the other modes are rare: hide the callee from inference, so that it doesn't
        # analyze the whole launch chain for each of them when compiling every kernel
        Base.inferencebarrier(launch_kernel)(obj, launch, ndrange, workgroupsize, iterspace, args)
    end
    return nothing
end

# Whether launching `iterspace` on a 1-D grid may take more groups than the grid of the
# backend holds (an N-d launch fits by construction, see `select_launch`). If the workgroup
# size will be tuned, `iterspace` is preliminary, and the number of groups is only known
# after compiling the kernel; a tuned workgroup has at least one work-item of the `ndrange`,
# so assume the worst, one group per work-item.
function exceeds_grid(obj::Kernel, workgroupsize, iterspace)
    tuned = KernelAbstractions.workgroupsize(obj) <: DynamicSize && workgroupsize === nothing
    groups = saturated_prod(size(tuned ? workitems(iterspace) : blocks(iterspace)))
    return groups > first(KI.max_num_groups(backend(obj)))
end

function launch_kernel(obj::Kernel, launch, ndrange, _workgroupsize, iterspace, args::Tuple)
    b = backend(obj)

    # this might not be the final context, since we may tune the workgroupsize
    ctx = mkcontext(obj, ndrange, iterspace, launch)
    args, tt = prepare_arguments(b, obj.f, ctx, args)
    kernel = KI.kernel_function(b, obj.f, tt; compiler_options(obj)...)

    # tune the workgroup size, keeping the context type (and thus the kernel) the same
    if workgroupsize(obj) <: DynamicSize && _workgroupsize === nothing
        range = something(ndrange, static_ndrange(obj))
        threads = KI.launch_configuration(kernel; nitems = saturated_prod(extents(range))).workgroupsize
        iterspace, _ = partition(obj, ndrange, launch_workgroupsize(b, launch, threads, range))
        ctx = mkcontext(obj, ndrange, iterspace, launch)
    end

    # the geometry is valid by construction, except for the kernel's limits
    groups = size(blocks(iterspace))
    items = size(workitems(iterspace))
    if launch isa NDLaunch
        groups, items = KI.pad3(groups), KI.pad3(items)
    else
        groups, items = (prod(groups), 1, 1), (prod(items), 1, 1)
    end
    KI.check_work_group_size(kernel, items)
    KI.launch(kernel, groups, items, prepend(ctx, args))
    return nothing
end

## grid-stride launches

# the number of groups a grid-stride launch may use, `nothing` to let the backend choose
check_gridstride(gridstride::Bool) = nothing
function check_gridstride(gridstride::Integer)
    gridstride >= 1 || throw(ArgumentError("`gridstride` must be `true`, `false` or a positive number of workgroups, got $gridstride"))
    return Int(gridstride)
end
check_gridstride(gridstride) =
    throw(ArgumentError("`gridstride` must be `true`, `false` or a positive number of workgroups, got $(repr(gridstride))"))

"""
    GridStride{Sync}(f)

The kernel function of a grid-stride launch: every hardware group calls the kernel
function `f` for the groups of the iteration space whose (0-based) linear index is
congruent to its own modulo the number of hardware groups, with a context whose
[`GridStrideLaunch`](@ref) holds that index. If `Sync`, the kernel may synchronize its
workgroup, and the groups synchronize before every next call, so that the work-items of
the call don't overwrite local memory that those of the previous one still read.

The number of calls is the same for all work-items of a group, which keeps a
`@synchronize` in `f` legal, and `f` returns from its call, not the loop, so that
`return` in the kernel body skips the rest of the body for the current index only.
"""
struct GridStride{Sync, F}
    f::F
    GridStride{Sync}(f::F) where {Sync, F} = new{Sync, F}(f)
end
Adapt.adapt_structure(to, k::GridStride{Sync}) where {Sync} = GridStride{Sync}(Adapt.adapt(to, k.f))

@inline with_launch(cm::CompilerMetadata{NDRange, CB, Nothing}, launch) where {NDRange, CB} =
    CompilerMetadata{NDRange, CB}(cm.ndrange, cm.iterspace; launch)

@inline function (k::GridStride{Sync})(ctx::CompilerMetadata, args::Vararg{Any, N}) where {Sync, N}
    launch = __launch(ctx)
    T = index_type(launch)
    n = length(blocks(__iterspace(ctx))) % T
    stride = KI.get_num_groups(T).x
    group = KI.get_group_id(T).x - one(T)
    # the host launches at most `n` groups, so `n - stride` doesn't wrap; comparing to it
    # rather than computing `group + stride` keeps the loop from wrapping either
    NDIteration.assume(stride <= n)
    NDIteration.assume(group < stride)
    while true
        @inline k.f(with_launch(ctx, typeof(launch)(group)), args...)
        group >= n - stride && break
        Sync && KI.barrier()
        group += stride
    end
    return nothing
end

# Whether `gridstride = true` launches fewer groups than the iteration space has on
# `backend`. On PoCL it doesn't: PoCL has no limit on the number of groups, it already hands
# the groups to its threads as they become idle, which is what a grid-stride loop does on a
# GPU, and a grid-stride loop keeps it from vectorizing over the work-items, which makes
# memory-bound kernels 2-4x slower (see `gridstride_barrier`).
occupancy_gridstride(backend) = !(backend isa POCL.POCLBackend)

# Whether a grid-stride launch on `backend` synchronizes the workgroup between the groups of
# the iteration space even if the kernel doesn't need it. PoCL runs a workgroup on a single
# thread, looping over its work-items between barriers: without one, it loops over the
# work-items around the whole grid-stride loop, so that every work-item walks the iteration
# space by itself, with a stride of the number of work-items launched, which isn't
# vectorized and is up to 40x slower for memory-bound kernels. A barrier puts the loop over
# the work-items inside the grid-stride loop instead, and costs PoCL next to nothing.
gridstride_barrier(backend) = backend isa POCL.POCLBackend

# Whether kernel function `f` may synchronize its workgroup when called with `args`.
@inline @generated function kernel_may_synchronize(backend, f, args::Tuple)
    n = fieldcount(args)
    converted = [Symbol(:arg, i) for i in 1:n]
    conversions = (:($(converted[i]) = KI.argconvert(backend, args[$i])) for i in 1:n)
    return quote
        $(conversions...)
        return f(SyncQuery(), $(converted...)) === Val(true)
    end
end

# The name of the kernel function `F` compiled for the argument types `TT`, which the kernel
# of a grid-stride launch takes rather than that of the wrapper, as without a stride. Mangling
# it takes longer than the rest of a launch, so do it once per signature.
@generated kernel_name(::Type{F}, ::Type{TT}) where {F, TT} =
    GPUCompiler.mangle_sig(Tuple{F, TT.parameters...})

# Launch `obj` on at most `maxgroups` groups (`nothing`: about as many as can be active at
# once), and at most as many as the grid of the backend holds, each looping over a share of
# the groups of the iteration space.
function launch_gridstride(obj::Kernel, maxgroups, ndrange, _workgroupsize, iterspace, args::Tuple)
    b = backend(obj)

    # the work-items of a group are launched as without a grid stride, and the index type
    # has to hold the padded iteration space, so choose as for a launch without one. the
    # groups are launched on a 1-D grid, which is at most as large as the number of groups
    # of the iteration space, so its limits on the number of groups don't matter
    inner = select_launch(obj, _workgroupsize, iterspace)
    T = index_type(inner)
    launch = GridStrideLaunch{T, typeof(inner)}(zero(T))

    ctx = mkcontext(obj, ndrange, iterspace, launch)
    args, tt = prepare_arguments(b, obj.f, ctx, args)
    f = GridStride{gridstride_barrier(b) || kernel_may_synchronize(b, obj.f, args)}(obj.f)
    kernel = KI.kernel_function(b, f, tt; name = kernel_name(Core.Typeof(obj.f), tt), compiler_options(obj)...)

    if workgroupsize(obj) <: DynamicSize && _workgroupsize === nothing
        range = something(ndrange, static_ndrange(obj))
        threads = KI.launch_configuration(kernel; nitems = saturated_prod(extents(range))).workgroupsize
        iterspace, _ = partition(obj, ndrange, launch_workgroupsize(b, inner, threads, range))
        ctx = mkcontext(obj, ndrange, iterspace, launch)
    end

    items = size(workitems(iterspace))
    items = inner isa NDLaunch ? KI.pad3(items) : (prod(items), 1, 1)
    nblocks = length(blocks(iterspace))
    if maxgroups === nothing
        maxgroups = KI.max_active_groups(kernel, prod(items))
        # unknown: one group per group of the iteration space, i.e. no stride
        maxgroups = maxgroups > 0 ? maxgroups : nblocks
    end
    groups = (min(nblocks, maxgroups, first(KI.max_num_groups(b))), 1, 1)

    KI.check_work_group_size(kernel, items)
    KI.launch(kernel, groups, items, prepend(ctx, args))
    return nothing
end

# The helpers below avoid splatting the arguments, and `map`, which isn't type stable for 32
# or more elements.

# Wrap the arguments the kernel marks `@Const` in a `ConstArg`, and return them together
# with the types of the converted arguments, which the kernel is compiled for. Converting a
# `ConstArg` constifies the converted argument, so each argument only needs to be converted
# once, to find the kernel method's `@Const` mask and to compute its type.
@inline @generated function prepare_arguments(backend, f, ctx, args::Tuple)
    n = fieldcount(args)
    converted = [Symbol(:arg, i) for i in 1:n]
    conversions = (:($(converted[i]) = KI.argconvert(backend, args[$i])) for i in 1:n)
    return quote
        converted_ctx = KI.argconvert(backend, ctx)
        $(conversions...)
        mask = f(ConstArgs(), $(converted...))
        return mark_constargs(mask, args),
            argument_types(mask, converted_ctx, ($(converted...),))
    end
end

# a kernel with varargs can have more arguments than its mask
is_constarg(mask, i) = i <= length(mask) && mask[i]

# no argument is marked, or the generator of a generated kernel failed, which compiling it
# reports
mark_constargs(_, args::Tuple) = args
@inline @generated function mark_constargs(::Val{mask}, args::Tuple) where {mask}
    marked = (is_constarg(mask, i) ? :(ConstArg(args[$i])) : :(args[$i]) for i in 1:fieldcount(args))
    return :(($(marked...),))
end

# `Tuple{map(Core.Typeof, (ctx, args...))...}`, with the `@Const` arguments constified
@inline @generated function argument_types(mask, ctx, args::Tuple)
    marked = mask <: Val ? mask.parameters[1] : ()
    types = (
        is_constarg(marked, i) ? :(Core.Typeof(constify(args[$i]))) : :(Core.Typeof(args[$i]))
            for i in 1:fieldcount(args)
    )
    return :(Tuple{Core.Typeof(ctx), $(types...)})
end

# `(ctx, args...)`
@inline @generated function prepend(ctx, args::Tuple)
    argexprs = (:(args[$i]) for i in 1:fieldcount(args))
    return :((ctx, $(argexprs...)))
end
