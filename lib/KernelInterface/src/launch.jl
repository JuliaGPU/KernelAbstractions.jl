# host-side operations related to kernel launches

"""
    Kernel{Backend, Kern}

A kernel compiled by [`kernel_function`](@ref) for `backend`, wrapping the backend's own
kernel object `kern`. `kernel.backend` is the backend value that was passed to
`kernel_function`.

Calling a `Kernel` launches it:

    (kernel::Kernel)(args...; numgroups=(), workgroupsize=(), ndrange=(),
                     max_work_group_size=typemax(Int), kwargs...)

`args` are the host-side arguments, e.g. a `CuArray` rather than a `CuDeviceArray`. The
backend converts them with [`argconvert`](@ref), and the converted types have to match the
argument types the kernel was compiled for.

The launch geometry is given in one of three ways:

- `numgroups` and `workgroupsize`: launch exactly that many work-groups of that many
  work-items. Either defaults to 1.
- `ndrange` and `workgroupsize`: launch `cld.(ndrange, workgroupsize)` work-groups.
- `ndrange` alone: the work-group size is chosen with [`launch_configuration`](@ref), bounded
  by `max_work_group_size` and by [`max_work_group_dims`](@ref), and filled first dimension
  first.

Each is an `Integer` or a tuple of up to 3 `Integer`s; missing dimensions are 1. `ndrange`
and `numgroups` are mutually exclusive.

`ndrange` is rounded up to whole work-groups and is not masked: the kernel runs for every
work-item of every launched group, and [`get_global_size`](@ref) returns the padded size.
Kernels have to check their own bounds.

A zero anywhere in `ndrange` or `numgroups` launches nothing. Work-group sizes must be
positive and fit [`max_work_group_dims`](@ref) and [`max_work_group_size`](@ref)`(kernel)`,
and the number of work-items in each dimension must fit an `Int`; anything else throws an
`ArgumentError` before the backend sees it. The number of work-groups is validated by the
backend.

Other keyword arguments are passed to [`launch`](@ref) unchanged. They are
backend-specific: a backend throws an error for keywords it doesn't support.

A launch is queued on the calling task's queue of the active device, and returns `nothing`.
"""
struct Kernel{B, Kern}
    backend::B
    kern::Kern
end

# `Vararg{Any, N}` makes Julia specialize on the arguments, which are only passed through
function (kernel::Kernel)(
        args::Vararg{Any, N}; numgroups = (), workgroupsize = (), ndrange = (),
        max_work_group_size::Integer = typemax(Int), kwargs...
    ) where {N}
    groups, items = launch_geometry(kernel, numgroups, workgroupsize, ndrange, max_work_group_size)
    any(iszero, groups) && return nothing
    launch(kernel, groups, items, args...; kwargs...)
    return nothing
end

"""
    launch(kernel::Kernel, groups::Dims{3}, items::Dims{3}, args...; kwargs...)

Launch `kernel` with `groups` work-groups of `items` work-items each, passing the host-side
arguments `args`. This is what calling a [`Kernel`](@ref) does after validating and
normalizing the launch geometry; users call the kernel instead.

`groups` and `items` are positive, `items` fits [`max_work_group_dims`](@ref) and
[`max_work_group_size`](@ref)`(kernel)`, and `groups .* items` doesn't overflow `Int`.
`kwargs` are the keyword arguments of the call that KernelInterface doesn't know.

!!! note
    Backend implementations **must** implement:
    ```
    launch(kernel::Kernel{<:NewBackend}, groups::Dims{3}, items::Dims{3}, args...; kwargs...)
    ```
    It converts `args` with [`argconvert`](@ref) (or lets its native launcher do so), and
    queues the launch on the calling task's queue; it doesn't have to wait for the kernel to
    complete. Declare the arguments as `args::Vararg{Any, N}` (with `where {N}`): Julia
    doesn't specialize a method on `args...` that it only passes through, which makes every
    launch dispatch dynamically. It must throw for keywords it does not support, and may
    throw for a number of work-groups the device cannot launch, or for a geometry that
    backend-specific compiler options of the kernel don't allow.
"""
function launch end

# the launch keywords are either scalars or tuples of up to 3 integers
const LaunchDims = Union{Integer, Tuple{}, NTuple{1, Integer}, NTuple{2, Integer}, NTuple{3, Integer}}

@inline pad3(x::Integer) = (Int(x), 1, 1)
@inline pad3(x::Tuple) = (map(Int, x)..., ntuple(_ -> 1, Val(3 - length(x)))...)

@noinline function throw_launch_error(name, value)
    throw(ArgumentError("`$name` must be an integer or a tuple of up to 3 integers, got $(repr(value))"))
end

@noinline function throw_range_error(name, value)
    throw(ArgumentError("`$name` must be between 0 and typemax(Int), got $(repr(value))"))
end

@inline function check_dims(name, x)
    x isa LaunchDims || throw_launch_error(name, x)
    any(d -> d < 0 || d > typemax(Int), x) && throw_range_error(name, x)
    return
end

# the product of positive `dims`, or `cap` if it is larger, without overflowing
@inline function capped_prod(dims::Dims, cap::Int)
    p = 1
    for d in dims
        d > cap ÷ p && return cap
        p *= d
    end
    return p
end

# whether the product of positive `dims` exceeds `limit`, without overflowing
@inline function prod_exceeds(dims::Dims, limit::Int)
    p = 1
    for d in dims
        d > limit ÷ p && return true
        p *= d
    end
    return false
end

"""
    launch_geometry(kernel, numgroups, workgroupsize, ndrange, max_work_group_size)::Tuple{Dims{3}, Dims{3}}

Validate the launch keywords of a [`Kernel`](@ref) call and turn them into the number of
work-groups and the work-group size. A zero number of work-groups means nothing is launched.
"""
@inline function launch_geometry(kernel::Kernel, numgroups, workgroupsize, ndrange, max_work_group_size)
    check_dims("numgroups", numgroups)
    check_dims("workgroupsize", workgroupsize)
    check_dims("ndrange", ndrange)
    if ndrange != () && numgroups != ()
        throw(ArgumentError("Only one of `numgroups` and `ndrange` can be used"))
    end
    max_work_group_size > 0 ||
        throw(ArgumentError("`max_work_group_size` must be positive, got $max_work_group_size"))

    items = if workgroupsize != ()
        wgsize = pad3(workgroupsize)
        any(iszero, wgsize) &&
            throw(ArgumentError("`workgroupsize` must be positive, got $(repr(workgroupsize))"))
        check_work_group_size(kernel, wgsize)
        wgsize
    elseif ndrange == ()
        (1, 1, 1)
    elseif any(iszero, ndrange)
        return (0, 0, 0), (1, 1, 1)
    else
        wanted = pad3(ndrange)
        config = launch_configuration(
            kernel; nitems = capped_prod(wanted, typemax(Int)),
            max_work_group_size = min(max_work_group_size, typemax(Int)) % Int
        )
        threads_to_workgroupsize(config.workgroupsize, wanted, max_work_group_dims(kernel.backend))
    end

    groups = if ndrange != ()
        cld.(pad3(ndrange), items)
    elseif numgroups != ()
        pad3(numgroups)
    else
        (1, 1, 1)
    end

    # the global size has to fit an `Int`, as `get_global_size()` returns it
    if !any(iszero, groups) && any(map((g, i) -> g > typemax(Int) ÷ i, groups, items))
        throw(ArgumentError("Launch of $groups work-groups of $items work-items has more than typemax(Int) work-items in a dimension"))
    end
    return groups, items
end

function check_work_group_size(kernel::Kernel, items::Dims{3})
    max_dims = max_work_group_dims(kernel.backend)
    all(items .<= max_dims) ||
        throw(ArgumentError("Work-group size $items exceeds the maximum of $max_dims per dimension"))
    max_items = max_work_group_size(kernel)
    prod_exceeds(items, max_items) &&
        throw(ArgumentError("Work-group size $items has more than $max_items work-items, the maximum for this kernel"))
    return
end

"""
    threads_to_workgroupsize(threads, ndrange, [limits])

Distribute `threads` work-items over the dimensions of `ndrange`, filling the first
dimension first. Dimension `d` gets at most `limits[d]` work-items; dimensions past the end
of `limits` are only bounded by `threads`.

Every dimension gets at least one work-item, even for a zero-sized `ndrange`.

Not part of the public interface; used by KernelInterface's and KernelAbstractions' launch
code.
"""
threads_to_workgroupsize(threads, ndrange::Tuple, limits = ()) =
    _threads_to_workgroupsize(threads, 1, ndrange, limits)
threads_to_workgroupsize(threads, ndrange::Integer, limits = ()) =
    only(threads_to_workgroupsize(threads, (ndrange,), limits))
# written recursively, because a closure updating the running total would box it
_threads_to_workgroupsize(threads, total, ::Tuple{}, limits) = ()
function _threads_to_workgroupsize(threads, total, ndrange::Tuple, limits)
    limit = isempty(limits) ? typemax(Int) : first(limits)
    x = max(min(div(threads, total), first(ndrange), limit), 1)
    rest = isempty(limits) ? () : Base.tail(limits)
    return (x, _threads_to_workgroupsize(threads, total * x, Base.tail(ndrange), rest)...)
end


## limits and advice

"""
    max_work_group_size(backend)::Int
    max_work_group_size(kernel::Kernel)::Int

The largest number of work-items a work-group can have: on the active device of `backend`,
or for launches of the compiled `kernel` (which may be lower, e.g. because of the kernel's
register use). Launching a larger work-group is an error.

The work-group size that performs best is often smaller; see [`launch_configuration`](@ref).

!!! note
    Backend implementations **must** implement both:
    ```
    max_work_group_size(backend::NewBackend)::Int
    max_work_group_size(kernel::Kernel{<:NewBackend})::Int
    ```
    The kernel form answers for the device the kernel was compiled for.
"""
function max_work_group_size end

"""
    launch_configuration(kernel::Kernel; nitems=nothing, max_work_group_size=typemax(Int))::@NamedTuple{workgroupsize::Int}

The recommended number of work-items per work-group for launching `kernel` over `nitems`
work-items in total (`nothing` if unknown), at most `max_work_group_size`. This is what an
`ndrange` launch without a `workgroupsize` uses, passing the number of work-items in the
`ndrange` (saturated at `typemax(Int)`). `nitems` and `max_work_group_size` are positive.

Unlike [`max_work_group_size`](@ref), this is advice: backends may base it on occupancy or
on the size of the launch, e.g. to prefer more work-groups over larger ones.

!!! note
    Backend implementations **may** implement:
    ```
    launch_configuration(kernel::Kernel{<:NewBackend}; nitems::Union{Int, Nothing}=nothing,
                         max_work_group_size::Int=typemax(Int))::@NamedTuple{workgroupsize::Int}
    ```
    The result has to be positive and at most `max_work_group_size` and
    `max_work_group_size(kernel)`. The fallback recommends the largest legal work-group
    size.
"""
function launch_configuration(
        kernel::Kernel; nitems::Union{Integer, Nothing} = nothing,
        max_work_group_size::Integer = typemax(Int)
    )
    return (; workgroupsize = Int(min(KernelInterface.max_work_group_size(kernel), max_work_group_size)))
end

"""
    max_work_group_dims(backend)::NTuple{3, Int}

The maximum number of work-items along each dimension of a work-group, for the active
device of `backend`. [`max_work_group_size`](@ref) bounds their product.

!!! note
    Backend implementations **must** implement:
    ```
    max_work_group_dims(backend::NewBackend)::NTuple{3, Int}
    ```
"""
function max_work_group_dims end

"""
    max_num_groups(backend)::NTuple{3, Int}

The maximum number of work-groups along each dimension of a launch, for the active device
of `backend`.

This is conservative: a launch within these limits works for any work-group size (as long
as the number of work-items in each dimension fits an `Int`), but some backends accept more
work-groups for smaller work-groups (e.g. HIP bounds the number of work-items per
dimension). The backend's validation at launch time is authoritative.

!!! note
    Backend implementations **must** implement:
    ```
    max_num_groups(backend::NewBackend)::NTuple{3, Int}
    ```
"""
function max_num_groups end

"""
    sub_group_size(backend)::Int

The sub-group width of kernels compiled for the active device of `backend`.

Kernels compiled by [`kernel_function`](@ref) execute with exactly this width: on the device,
[`get_max_sub_group_size`](@ref) returns it, and full sub-groups have this many work-items.
Host code can rely on it, e.g. to pick a `Val(N)` for a warp-level reduction.

!!! note
    Backend implementations **must** implement this if [`supports_subgroups`](@ref) returns
    `true`:
    ```
    sub_group_size(backend::NewBackend)::Int
    ```
    A backend that cannot guarantee the width for every kernel has to report
    `supports_subgroups(backend) = false`.
"""
function sub_group_size end

"""
    multiprocessor_count(backend::NewBackend)::Int

The multiprocessor count for the current device used by `backend`.
Used for certain algorithm optimizations.

!!! note
    Backend implementations **may** implement:
    ```
    multiprocessor_count(backend::NewBackend)::Int
    ```
    As well as the on-device functionality.
"""
multiprocessor_count(::Backend) = 0

"""
    argconvert(::NewBackend, arg)

This function is called for every argument to be passed to a kernel,
converting them to their device side representation.

!!! note
    Backend implementations **must** implement:
    ```
    argconvert(::NewBackend, arg)
    ```
"""
function argconvert end

"""
    kernel_function(backend, f::F, tt::TT=Tuple{}; name=nothing, kwargs...)::Kernel

Compile the function `f` for arguments of the (device-side) types `tt`, for the active
device of `backend`, returning a [`Kernel`](@ref). For a higher-level interface, use
[`KernelInterface.@launch`](@ref).

Keyword arguments:
- `name`: override the name that the kernel will have in the generated code.

Other keyword arguments are backend-specific compiler options (e.g. `maxthreads` for
CUDA.jl); backends throw an error for options they don't support.

!!! note
    Backend implementations **must** implement:
    ```
    kernel_function(backend::NewBackend, f::F, tt::TT=Tuple{}; name=nothing, kwargs...) where {F,TT}
    ```
    Kernels must execute with sub-group width [`sub_group_size(backend)`](@ref sub_group_size)
    if the backend supports sub-groups.
"""
function kernel_function end

const MACRO_KWARGS = [:launch]
const LAUNCH_KWARGS = [:numgroups, :workgroupsize, :ndrange, :max_work_group_size]

"""
    KI.@launch backend [launch=true] [numgroups=...] [workgroupsize=...] [ndrange=...] [max_work_group_size=...] [kwargs...] f(args...)

Compile `f(args...)` for `backend` and launch it, like `@cuda` or `@metal` do.

`f` and the arguments are converted with [`argconvert`](@ref) and compiled with
[`kernel_function`](@ref), and the resulting [`Kernel`](@ref) is called with the launch
keywords `numgroups`, `workgroupsize`, `ndrange` and `max_work_group_size`, whose meaning
is documented there. The arguments are kept alive while the launch is being queued.

Other keyword arguments:
- `launch`: whether to launch the kernel, defaults to `true`. With `launch=false`, the
  kernel is only compiled and returned, and the launch keywords can't be used: launch it by
  calling it with the arguments and the launch keywords.
- `name` and any other keyword are passed to [`kernel_function`](@ref) as compiler options.

Launch options specific to a backend (such as a CUDA stream) can't be passed to
`@launch`; use `launch=false` and pass them when calling the kernel.

`backend` is evaluated once. Returns the `Kernel`.

```julia
function vadd(c, a, b)
    i = KI.get_global_id().x
    if i <= length(c)
        @inbounds c[i] = a[i] + b[i]
    end
    return
end

KI.@launch backend ndrange=length(c) vadd(c, a, b)
```
"""
macro launch(backend, ex...)
    isempty(ex) && throw(ArgumentError("KI.@launch needs a function call to launch"))
    call = ex[end]
    kwargs = map(ex[1:(end - 1)]) do kwarg
        if kwarg isa Symbol
            :($kwarg = $kwarg)
        elseif Meta.isexpr(kwarg, :(=))
            kwarg
        else
            throw(ArgumentError("Invalid keyword argument '$kwarg'"))
        end
    end

    # destructure the kernel call
    Meta.isexpr(call, :call) || throw(ArgumentError("final argument to KI.@launch should be a function call"))
    f = call.args[1]
    args = call.args[2:end]

    code = quote end
    vars, var_exprs = assign_args!(code, args)

    # group keyword argument; everything we don't know is a compiler option
    macro_kwargs, call_kwargs, compiler_kwargs =
        split_kwargs(kwargs, MACRO_KWARGS, LAUNCH_KWARGS)

    # handle keyword arguments that influence the macro's behavior
    launch = true
    for kwarg in macro_kwargs
        key, val = kwarg.args
        isa(val, Bool) || throw(ArgumentError("`launch` keyword argument to KI.@launch should be a Bool"))
        launch = val::Bool
    end
    if !launch && !isempty(call_kwargs)
        throw(ArgumentError("KI.@launch with launch=false does not support launch keyword arguments; use them when calling the kernel"))
    end

    # FIXME: macro hygiene wrt. escaping kwarg values (this broke with 1.5)
    #        we esc() the whole thing now, necessitating gensyms...
    @gensym backend_var f_var kernel_f kernel_args kernel_tt kernel

    # convert the arguments, call the compiler and launch the kernel
    # while keeping the original arguments alive
    push!(
        code.args,
        quote
            $backend_var = $backend
            $f_var = $f
            GC.@preserve $(vars...) $f_var begin
                $kernel_f = $argconvert($backend_var, $f_var)
                $kernel_args = Base.map(x -> $argconvert($backend_var, x), ($(var_exprs...),))
                $kernel_tt = Tuple{Base.map(Core.Typeof, $kernel_args)...}
                $kernel = $kernel_function($backend_var, $kernel_f, $kernel_tt; $(compiler_kwargs...))
                if $launch
                    $kernel($(var_exprs...); $(call_kwargs...))
                end
                $kernel
            end
        end
    )

    return esc(
        quote
            let
                $code
            end
        end
    )
end
