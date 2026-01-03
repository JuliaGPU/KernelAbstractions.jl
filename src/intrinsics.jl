"""
# `KernelIntrinsics`

The `KernelIntrinsics` (or `KI`) module defines the API interface for backends to define various lower-level device and
host-side functionality. The `KI` intrinsics are used to define the higher-level device-side
intrinsics functionality in `KernelAbstractions`.

Both provide APIs for host and device-side functionality, but `KI` focuses on on lower-level
functionality that is shared amongst backends, while `KernelAbstractions` provides higher-level functionality
such as writing kernels that work on arrays with an arbitrary number of dimensions, or convenience functions
like allocating arrays on a backend.
"""
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
    get_sub_group_size()::UInt32

Returns the number of work-items in the sub-group.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_sub_group_size()::UInt32
    ```
"""
function get_sub_group_size end

"""
    get_max_sub_group_size()::UInt32

Returns the maximum sub-group size for sub-groups in the current workgroup.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_max_sub_group_size()::UInt32
    ```
"""
function get_max_sub_group_size end

"""
    get_num_sub_groups()::UInt32

Returns the number of sub-groups in the current workgroup.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_num_sub_groups()::UInt32
    ```
"""
function get_num_sub_groups end

"""
    get_sub_group_id()::UInt32

Returns the sub-group ID within the work-group.

!!! note
    1-based.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_sub_group_id()::UInt32
    ```
"""
function get_sub_group_id end

"""
    get_sub_group_local_id()::UInt32

Returns the work-item ID within the current sub-group.

!!! note
    1-based.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_sub_group_local_id()::UInt32
    ```
"""
function get_sub_group_local_id end


"""
    localmemory(::Type{T}, dims)

Declare memory that is local to a workgroup.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override localmemory(::Type{T}, ::Val{Dims}) where {T, Dims}
    ```
    As well as the on-device functionality.
"""
localmemory(::Type{T}, dims) where {T} = localmemory(T, Val(dims))

"""
    shfl_down(val::T, offset::Integer) where T

Read `val` from a lane with higher id given by `offset`.
When writing kernels using this function, it should be
assumed that it is not synchronized.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override shfl_down(val::T, offset::Integer) where T
    ```
    As well as the on-device functionality.
"""
function shfl_down end

"""
    shfl_down_types(::Backend)::Vector{DataType}

Returns a vector of `DataType`s supported on `backend`

!!! note
    Backend implementations **must** implement this function
    only if they support `shfl_down` for any types.
"""
shfl_down_types(::Backend) = DataType[]


"""
    barrier()

After a `barrier()` call, all read and writes to global and local memory
from each thread in the workgroup are visible in from all other threads in the
workgroup.

This does **not** guarantee that a write from a thread in a certain workgroup will
be visible to a thread in a different workgroup.

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
    sub_group_barrier()

After a `sub_group_barrier()` call, all read and writes to global and local memory
from each thread in the sub-group are visible in from all other threads in the
sub-group.

This does **not** guarantee that a write from a thread in a certain sub-group will
be visible to a thread in a different sub-group.

!!! note
    `sub_group_barrier()` must be encountered by all workitems of a sub-group executing the kernel or by none at all.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override sub_group_barrier()
    ```
"""
function sub_group_barrier()
    error("Sub-group barrier used outside kernel or not captured")
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
    Kernel{Backend, Kern}

Kernel closure struct that is used to represent the backend
kernel on the host.

!!! note
    Backend implementations **must** implement:
    ```
    (kernel::Kernel{<:NewBackend})(args...; numworkgroups=1, workgroupsize=1)
    ```
    With the `numworkgroups` and `workgroupsize` arguments accepting a scalar Integer
    or or a 1, 2, or 3 Integer tuple and throwing an `ArgumentError` otherwise. The
    helper function `KI.check_launch_args(numworkgroups, workgroupsize)` can be used
    by the backend or a custom check can be implemented.

    Backends must also implement the on-device kernel launch functionality.
"""
struct Kernel{B,Kern}
    backend::B
    kern::Kern
end

function check_launch_args(numworkgroups, workgroupsize)
    length(numworkgroups) <= 3 ||
        throw(ArgumentError("`numworkgroups` only accepts up to 3 dimensions"))
    length(workgroupsize) <= 3 ||
        throw(ArgumentError("`workgroupsize` only accepts up to 3 dimensions"))
    return
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
multiprocessor_count(_) = 0

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
a callable kernel object. For a higher-level interface, use [`KI.@kernel`](@ref).

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
    kwargs = map(ex[1:(end-1)]) do kwarg
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

end
