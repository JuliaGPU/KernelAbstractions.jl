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
tuned, the iteration space is preliminary: it uses the `ndrange` as the workgroup size.
"""
function launch_config(kernel::Kernel, _ndrange, _workgroupsize)
    if _ndrange isa Integer
        _ndrange = (_ndrange,)
    end
    if _workgroupsize isa Integer
        _workgroupsize = (_workgroupsize,)
    end

    iterspace, dynamic = if workgroupsize(kernel) <: DynamicSize && _workgroupsize === nothing
        # use the ndrange as preliminary workgroupsize for autotuning
        partition(kernel, _ndrange, something(_ndrange, static_ndrange(kernel)))
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

function launch_tuple(obj::Kernel, args::Tuple; ndrange = nothing, workgroupsize = nothing)
    profiling_active() || return launch_untraced(obj, args, ndrange, workgroupsize)
    # The traced launch calls the same `launch_untraced`, so that it isn't inferred twice for
    # every kernel. The helpers around it are only inferred once.
    id = start_launch_range(obj)
    try
        launch_untraced(obj, args, ndrange, workgroupsize)
        # a profiler that measures kernels rather than launches
        synchronizes_launches(id) && KI.synchronize(backend(obj))
    finally
        profiling_range_end(id)
    end
    return nothing
end

Base.@nospecializeinfer @noinline start_launch_range(@nospecialize(obj::Kernel)) =
    profiling_range_start(kernel_label(obj.f))

function launch_untraced(obj::Kernel, args::Tuple, ndrange, workgroupsize)
    ndrange, workgroupsize, iterspace, dynamic = launch_config(obj, ndrange, workgroupsize)
    # nothing to launch (or compile) for an empty ndrange
    any(iszero, size(blocks(iterspace))) && return nothing

    # launch on an N-d grid, computing indices in 32 bits, if possible. this doesn't depend
    # on the tuned workgroup size, so the context (and thus the kernel) doesn't either.
    launch = select_launch(obj, workgroupsize, iterspace)
    if launch === NDLaunch{Int32}()
        # the common case, specialized statically
        launch_kernel(obj, NDLaunch{Int32}(), ndrange, workgroupsize, iterspace, args)
    else
        launch_kernel(obj, launch, ndrange, workgroupsize, iterspace, args)
    end
    return nothing
end

function launch_kernel(obj::Kernel, launch, ndrange, _workgroupsize, iterspace, args::Tuple)
    b = backend(obj)

    # this might not be the final context, since we may tune the workgroupsize
    ctx = mkcontext(obj, ndrange, iterspace, launch)
    kernel = compile(obj, ctx, args)

    # tune the workgroup size, keeping the context type (and thus the kernel) the same
    if workgroupsize(obj) <: DynamicSize && _workgroupsize === nothing
        range = something(ndrange, static_ndrange(obj))
        threads = KI.launch_configuration(kernel; nitems = saturated_prod(extents(range))).workgroupsize
        iterspace, _ = partition(obj, ndrange, launch_workgroupsize(b, launch, threads, range))
        ctx = mkcontext(obj, ndrange, iterspace, launch)
    end

    # launching through the `KI.Kernel` validates the sizes against the kernel's limits
    groups = size(blocks(iterspace))
    items = size(workitems(iterspace))
    if launch isa NDLaunch
        call_kernel(kernel, ctx, args, groups, items)
    else
        call_kernel(kernel, ctx, args, prod(groups), prod(items))
    end
    return nothing
end

@inline function compile(obj::Kernel, ctx, args::Tuple)
    b = backend(obj)
    tt = argument_types(b, ctx, args)
    return KI.kernel_function(b, obj.f, tt; compiler_options(obj)...)
end

# The helpers below avoid splatting the arguments, and `map`, which isn't type stable for 32
# or more elements.

# `Tuple{map(x -> Core.Typeof(KI.argconvert(backend, x)), (ctx, args...))...}`
@inline @generated function argument_types(backend, ctx, args::Tuple)
    types = (:(Core.Typeof(KI.argconvert(backend, args[$i]))) for i in 1:fieldcount(args))
    return :(Tuple{Core.Typeof(KI.argconvert(backend, ctx)), $(types...)})
end

# `kernel(ctx, args...; numgroups, workgroupsize)`
@inline @generated function call_kernel(kernel::KI.Kernel, ctx, args::Tuple, numgroups, workgroupsize)
    argexprs = (:(args[$i]) for i in 1:fieldcount(args))
    return :(kernel(ctx, $(argexprs...); numgroups, workgroupsize))
end
