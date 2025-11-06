module KernelIntrinsics

import ..KernelAbstractions: Backend
import GPUCompiler: split_kwargs, assign_args!

"""
    get_global_size()::@NamedTuple{x::Int, y::Int, z::Int}

Return the number of global work-items specified.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_global_size()::@NamedTuple{x::Int, y::Int, z::Int}
    ```
"""
function get_global_size end

"""
    get_global_id()::@NamedTuple{x::Int, y::Int, z::Int}

Returns the unique global work-item ID.

!!! note
    1-based.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_global_id()::@NamedTuple{x::Int, y::Int, z::Int}
    ```
"""
function get_global_id end

"""
    get_local_size()::@NamedTuple{x::Int, y::Int, z::Int}

Return the number of local work-items specified.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_local_size()::@NamedTuple{x::Int, y::Int, z::Int}
    ```
"""
function get_local_size end

"""
    get_local_id()::@NamedTuple{x::Int, y::Int, z::Int}

Returns the unique local work-item ID.

!!! note
    1-based.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_local_id()::@NamedTuple{x::Int, y::Int, z::Int}
    ```
"""
function get_local_id end

"""
    get_num_groups()::@NamedTuple{x::Int, y::Int, z::Int}

Returns the number of groups.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_num_groups()::@NamedTuple{x::Int, y::Int, z::Int}
    ```
"""
function get_num_groups end

"""
    get_group_id()::@NamedTuple{x::Int, y::Int, z::Int}

Returns the unique group ID.

!!! note
    1-based.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_group_id()::@NamedTuple{x::Int, y::Int, z::Int}
    ```
"""
function get_group_id end

"""
    localmemory(T, dims)

Declare memory that is local to a workgroup.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override localmemory(T::DataType, ::Val{Dims}) where {T, Dims}
    ```
    As well as the on-device functionality.
"""
localmemory(::Type{T}, dims) where {T} = localmemory(T, Val(dims))

"""
    barrier()

After a `barrier()` call, all read and writes to global and local memory
from each thread in the workgroup are visible in from all other threads in the
workgroup.

!!! note
    `barrier()` must be encountered by all workitems of a work-group executing the kernel or by none at all.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override barrier()
    ```
"""
function barrier()
    error("Group barrier used outside kernel or not captured")
end

"""
    _print(args...)

    Overloaded by backends to enable `KernelAbstractions.@print`
    functionality.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override _print(args...)
    ```
    If the backend does not support printing,
    define it to return `nothing`.
"""
function _print end



"""
    Kernel{Backend, BKern}

Kernel closure struct that is used to represent the backend
kernel on the host.

!!! note
    Backend implementations **must** implement:
    ```
    (kernel::Kernel{<:NewBackend})(args...; numworkgroups=nothing, workgroupsize=nothing, kwargs...)
    ```
    As well as the on-device functionality.
"""
struct Kernel{B, Kern}
    backend::B
    kern::Kern
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
multiprocessor_count(_) = 0

"""
    argconvert(::NewBackend, arg)

This function is called for every argument to be passed to a kernel, allowing it to be
converted to a GPU-friendly format.

!!! note
    Backend implementations **must** implement:
    ```
    argconvert(::NewBackend, arg)
    ```
"""
function argconvert end

"""
    KI.kifunction(::NewBackend, f::F, tt::TT=Tuple{}; name=nothing, kwargs...) where {F,TT}

Low-level interface to compile a function invocation for the currently-active GPU, returning
a callable kernel object. For a higher-level interface, use [`@kikernel`](@ref).

Currently, only `kifunction` only supports the `name` keyword argument as it is the only one
by all backends.

Keyword arguments:
- `name`: override the name that the kernel will have in the generated code

!!! note
    Backend implementations **must** implement:
    ```
    kifunction(::NewBackend, f::F, tt::TT=Tuple{}; name=nothing, kwargs...) where {F,TT}
    ```
"""
function kifunction end

const MACRO_KWARGS = [:launch]
const COMPILER_KWARGS = [:name]
const LAUNCH_KWARGS = [:numworkgroups, :workgroupsize]

"""
    @kikernel backend workgroupsize=... numworkgroups=... [kwargs...] func(args...)

High-level interface for executing code on a GPU.

The `@kikernel` macro should prefix a call, with `func` a callable function or object that
should return nothing. It will be compiled to a function native to the specified `backend`
upon first use, and to a certain extent arguments will be converted and managed automatically
using `argconvert`. Finally, if `launch=true`, the newly created callable kernel object is
called and launched according to the specified `backend`.

There are a few keyword arguments that influence the behavior of `@kikernel`:

- `launch`: whether to launch this kernel, defaults to `true`. If `false`, the returned
  kernel object should be launched by calling it and passing arguments again.
- `name`: the name of the kernel in the generated code. Defaults to an automatically-
  generated name.
"""
macro kikernel(backend, ex...)
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
    Meta.isexpr(call, :call) || throw(ArgumentError("final argument to @kikern should be a function call"))
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
            isa(val, Bool) || throw(ArgumentError("`launch` keyword argument to @kikern should be a Bool"))
            launch = val::Bool
        else
            throw(ArgumentError("Unsupported keyword argument '$key'"))
        end
    end
    if !launch && !isempty(call_kwargs)
        error("@kikern with launch=false does not support launch-time keyword arguments; use them when calling the kernel")
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
                $kernel = $kifunction($backend, $kernel_f, $kernel_tt; $(compiler_kwargs...))
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

end
