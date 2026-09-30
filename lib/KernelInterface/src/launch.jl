# host-side operations related to kernel launches

"""
    Kernel{Backend, Kern}

Kernel closure struct that is used to represent the backend
kernel on the host.

!!! note
    Backend implementations **must** implement:
    ```
    (kernel::Kernel{<:NewBackend})(args...; numgroups=(), workgroupsize=(), ndrange=(), max_work_group_size=typemax(Int))
    ```
    `numgroups`, `workgroupsize`, and `ndrange` must accept a scalar Integer, a 1, 2,
    or 3 Integer tuple, or an empty tuple. Otherwise, it must throw an `ArgumentError`. An
    `ArgumentError` must also be thrown if `ndrange` and `numgroups` are both specified.
    The helper function `KI.check_launch_args(numgroups, workgroupsize, ndrange)` can be
    used by the backend or a custom check can be implemented.

    `max_work_group_size` is to allow algorithms to request a max workgroupsize with `ndrange`.
    This is a maximum value because a kernel's maximum workitems per workgroup may be lower than
    requested.

    An `ndrange` with a zero-sized dimension, as when launching over an empty array, is
    not an error: the call must be a no-op and return `nothing` instead of launching.

    By default, kernels must launch with 1 workgroup containing 1 workitem.

    Backends must also implement the on-device kernel launch functionality.
"""
struct Kernel{B, Kern}
    backend::B
    kern::Kern
end

"""
    check_launch_args(numgroups, workgroupsize, ndrange)

Validate the launch configuration passed to a [`Kernel`](@ref), throwing an
`ArgumentError` if either argument has more than 3 dimensions, or if `ndrange`
and `numgroups` are both defined.

Backends may call this from their kernel-launch method instead of writing their
own check.
"""
function check_launch_args(numgroups, workgroupsize, ndrange)
    length(ndrange) > 0 && length(numgroups) > 0 &&
        throw(ArgumentError("Only one of `numgroups` and `ndrange` can be used"))
    length(numgroups) <= 3 ||
        throw(ArgumentError("`numgroups` only accepts up to 3 dimensions"))
    length(workgroupsize) <= 3 ||
        throw(ArgumentError("`workgroupsize` only accepts up to 3 dimensions"))
    length(ndrange) <= 3 ||
        throw(ArgumentError("`ndrange` only accepts up to 3 dimensions"))
    return
end

"""
    threads_to_workgroupsize(threads, ndrange, [limits])

Distribute `threads` work-items over the dimensions of `ndrange`, filling the first
dimension first. Dimension `d` gets at most `limits[d]` work-items; dimensions past the end
of `limits` are only bounded by `threads`.

Every dimension gets at least one work-item, even for a zero-sized `ndrange`.
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

"""
    auto_launch_sizes(kernel::KI.Kernel, numgroups, workgroupsize, ndrange, [max_work_items])

Returns a suggested `numgroups` and `workgroupsize` based on
the input arguments. This function assumes arguments have been
validated by `check_launch_args`.

If any `ndrange` dimension is zero, the returned `numgroups` is zero in
that dimension; backends should skip the launch in that case. Note that very
large `ndrange`s can produce total grid sizes >= 2^32, which is problematic
on some backends.

Backends may call this from their kernel-launch method instead of
writing their own heuristic for calculating launch size.
"""
@inline function auto_launch_sizes(kernel::Kernel, numgroups, workgroupsize, ndrange, max_work_items = typemax(Int))
    numgroups, workgroupsize = if ndrange == ()
        numgroups == () ? 1 : numgroups, workgroupsize == () ? 1 : workgroupsize
    else
        workgroupsize = if workgroupsize == ()
            max_wgs = kernel_max_work_group_size(kernel; max_work_items = min(prod(ndrange), max_work_items))
            threads_to_workgroupsize(max_wgs, ndrange, max_work_group_dims(kernel.backend))
        else
            workgroupsize
        end
        numgroups = cld.(ndrange, workgroupsize)
        Int.(numgroups), Int.(workgroupsize)
    end

    return numgroups, workgroupsize
end

"""
    kernel_max_work_group_size(kern; [max_work_items::Int])::Int

The maximum workgroup size limit for a kernel as reported by the backend.
This function should always be used to determine the workgroup size before
launching a kernel.

!!! note
    Backend implementations **must** implement:
    ```
    kernel_max_work_group_size(kern::Kernel{<:NewBackend}; max_work_items::Int=typemax(Int))::Int
    ```
    As well as the on-device functionality.
"""
function kernel_max_work_group_size end

"""
    max_work_group_size(backend, kern; [max_work_items::Int])::Int

The maximum workgroup size limit for a kernel as reported by the backend.
This function represents a theoretical maximum; `kernel_max_work_group_size`
should be used before launching a kernel as some backends may error if
kernel launch with too big a workgroup is attempted.

!!! note
    Backend implementations **must** implement:
    ```
    max_work_group_size(backend::NewBackend)::Int
    ```
    As well as the on-device functionality.
"""
function max_work_group_size end

"""
    max_work_group_dims(backend)::NTuple{3, Int}

The maximum number of work-items along each dimension of a workgroup, for the currently
active device of `backend`. [`max_work_group_size`](@ref) bounds their product.

!!! note
    Backend implementations **should** implement:
    ```
    max_work_group_dims(backend::NewBackend)::NTuple{3, Int}
    ```
    The fallback does not limit individual dimensions.
"""
max_work_group_dims(::Backend) = (typemax(Int), typemax(Int), typemax(Int))

"""
    max_num_groups(backend)::NTuple{3, Int}

The maximum number of workgroups along each dimension of a launch, for the currently
active device of `backend`.

!!! note
    Backend implementations **should** implement:
    ```
    max_num_groups(backend::NewBackend)::NTuple{3, Int}
    ```
    The fallback does not limit the number of workgroups.
"""
max_num_groups(::Backend) = (typemax(Int), typemax(Int), typemax(Int))

"""
    sub_group_size(backend)::Int

Returns a reasonable sub-group size supported by the currently
active device for the specified backend. This would typically
be 32, or 64 for devices that don't support 32.

!!! note
    Backend implementations **must** implement:
    ```
    sub_group_size(backend::NewBackend)::Int
    ```
    As well as the on-device functionality.
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
    KI.kernel_function(::NewBackend, f::F, tt::TT=Tuple{}; name=nothing, kwargs...) where {F,TT}

Low-level interface to compile a function invocation for the currently-active GPU, returning
a callable kernel object. For a higher-level interface, use
[`KernelInterface.@launch`](@ref).

Keyword arguments:
- `name`: override the name that the kernel will have in the generated code.

Other keyword arguments are backend-specific compiler options (e.g. `maxthreads` for
CUDA.jl); backends throw an error for options they don't support.

!!! note
    Backend implementations **must** implement:
    ```
    kernel_function(::NewBackend, f::F, tt::TT=Tuple{}; name=nothing, kwargs...) where {F,TT}
    ```
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
