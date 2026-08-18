# host-side operations related to kernel launches

"""
    Kernel{Backend, Kern}

Kernel closure struct that is used to represent the backend
kernel on the host.

!!! note
    Backend implementations **must** implement:
    ```
    (kernel::Kernel{<:NewBackend})(args...; numworkgroups=(), workgroupsize=(), ndrange=())
    ```
    `numworkgroups`, `workgroupsize`, and `ndrange` must accept a scalar Integer, a 1, 2,
    or 3 Integer tuple, or an empty tuple. Otherwise, it must throw an `ArgumentError`. An
    `ArgumentError` must also be thrown if `ndrange` and `numworkgroups` are both specified.
    The helper function `KI.check_launch_args(; numworkgroups, workgroupsize, ndrange)` can be
    used by the backend or a custom check can be implemented.

    By default, kernels must launch with 1 workgroup containing 1 workitem.

    Backends must also implement the on-device kernel launch functionality.
"""
struct Kernel{B, Kern}
    backend::B
    kern::Kern
end

"""
    check_launch_args(numworkgroups, workgroupsize, ndrange)

Validate the launch configuration passed to a [`Kernel`](@ref), throwing an
`ArgumentError` if either argument has more than 3 dimensions.

If valid, returns default values (empty tuple to 1)

Backends may call this from their kernel-launch method instead of writing their
own check.
"""
function check_launch_args(numworkgroups, workgroupsize, ndrange)
    length(ndrange) > 0 && length(numworkgroups) > 0 &&
        throw(ArgumentError("Only one of `numworkgroups` and `ndrange` can be used"))
    length(numworkgroups) <= 3 ||
        throw(ArgumentError("`numworkgroups` only accepts up to 3 dimensions"))
    length(workgroupsize) <= 3 ||
        throw(ArgumentError("`workgroupsize` only accepts up to 3 dimensions"))
    length(ndrange) <= 3 ||
        throw(ArgumentError("`ndrange` only accepts up to 3 dimensions"))
    return numworkgroups == () ? 1 : numworkgroups, workgroupsize == () ? 1 : workgroupsize, ndrange
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
[`KernelInterface.@kernel`](@ref).

Currently, `kernel_function` only supports the `name` keyword argument as it is the only one
by all backends.

Keyword arguments:
- `name`: override the name that the kernel will have in the generated code

!!! note
    Backend implementations **must** implement:
    ```
    kernel_function(::NewBackend, f::F, tt::TT=Tuple{}; name=nothing, kwargs...) where {F,TT}
    ```
"""
function kernel_function end

const MACRO_KWARGS = [:launch]
const COMPILER_KWARGS = [:name]
const LAUNCH_KWARGS = [:numworkgroups, :workgroupsize]

"""
    KI.@kernel backend workgroupsize=... numworkgroups=... [kwargs...] func(args...)

High-level interface for executing code on a GPU.

The `KI.@kernel` macro should prefix a call, with `func` a callable function or object that
should return nothing. It will be compiled to a function native to the specified `backend`
upon first use, and to a certain extent arguments will be converted and managed automatically
using `argconvert`. Finally, if `launch=true`, the newly created callable kernel object is
called and launched according to the specified `backend`.

There are a few keyword arguments that influence the behavior of `KI.@kernel`:

- `launch`: whether to launch this kernel, defaults to `true`. If `false`, the returned
  kernel object should be launched by calling it and passing arguments again.
- `name`: the name of the kernel in the generated code. Defaults to an automatically-
  generated name.

!!! note
    `KI.@kernel` differs from the `KernelAbstractions` macro in that this macro acts
    a wrapper around backend kernel compilation/launching (such as `@cuda`, `@metal`, etc.). It is
    used when calling a function to be run on a specific backend, while `KernelAbstractions.@kernel`
    is used kernel definition for use with the original higher-level `KernelAbstractions` API.
"""
macro kernel(backend, ex...)
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
    Meta.isexpr(call, :call) || throw(ArgumentError("final argument to KI.@kernel should be a function call"))
    f = call.args[1]
    args = call.args[2:end]

    code = quote end
    vars, var_exprs = assign_args!(code, args)

    # group keyword argument
    macro_kwargs, compiler_kwargs, call_kwargs, other_kwargs =
        split_kwargs(kwargs, MACRO_KWARGS, COMPILER_KWARGS, LAUNCH_KWARGS)
    if !isempty(other_kwargs)
        key, val = first(other_kwargs).args
        throw(ArgumentError("Unsupported keyword argument '$key'"))
    end

    # handle keyword arguments that influence the macro's behavior
    launch = true
    for kwarg in macro_kwargs
        key, val = kwarg.args
        if key === :launch
            isa(val, Bool) || throw(ArgumentError("`launch` keyword argument to KI.@kernel should be a Bool"))
            launch = val::Bool
        else
            throw(ArgumentError("Unsupported keyword argument '$key'"))
        end
    end
    if !launch && !isempty(call_kwargs)
        error("KI.@kernel with launch=false does not support launch-time keyword arguments; use them when calling the kernel")
    end

    # FIXME: macro hygiene wrt. escaping kwarg values (this broke with 1.5)
    #        we esc() the whole thing now, necessitating gensyms...
    @gensym f_var kernel_f kernel_args kernel_tt kernel

    # convert the arguments, call the compiler and launch the kernel
    # while keeping the original arguments alive
    push!(
        code.args,
        quote
            $f_var = $f
            GC.@preserve $(vars...) $f_var begin
                $kernel_f = $argconvert($backend, $f_var)
                $kernel_args = Base.map(x -> $argconvert($backend, x), ($(var_exprs...),))
                $kernel_tt = Tuple{Base.map(Core.Typeof, $kernel_args)...}
                $kernel = $kernel_function($backend, $kernel_f, $kernel_tt; $(compiler_kwargs...))
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
