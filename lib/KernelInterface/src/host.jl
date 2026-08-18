"""
    versioninfo(io::IO=stdout, backend::Backend)::Nothing

Print information about `backend` to `io`. It is up to the backends to
determine what is relevant.

!!! note
    Backend implementations **may** implement this function. If they do
    so, they should implement `versioninfo(io::IO, ::Backend)::Nothing`
"""
versioninfo(io::IO, b::Backend) = println(io, "`versioninfo` is not implemented for $b")
versioninfo(b::Backend) = versioninfo(stdout, b)

"""
    functional(::Backend)::Union{Bool, Missing}

Queries if the provided backend is functional. This may mean different
things for different backends, but generally should mean that the
necessary drivers and a compute device are available.

This function should return a `Bool` or `missing` if not implemented.

!!! compat "KernelAbstractions v0.9.22"
    This function was added in KernelAbstractions v0.9.22
"""
function functional(::Backend)
    return missing
end

"""
    synchronize(::Backend)

Synchronize the current backend.

!!! note
    Backend implementations **must** implement this function.
"""
function synchronize end


# Define:
#   adapt_storage(::Backend, a::Array) = adapt(BackendArray, a)
#   adapt_storage(::Backend, a::BackendArray) = a

"""
    priority!(::Backend, prio::Symbol)::Nothing

Set the priority for the backend stream/queue. This is an optional
feature that backends may or may not implement. If a backend shall
support priorities it must accept `:high`, `:normal`, `:low`.
Where `:normal` is the default.

!!! note
    Backend implementations **may** implement this function.
"""
function priority!(::Backend, prio::Symbol)
    if !(prio in (:high, :normal, :low))
        error("priority must be one of :high, :normal, :low")
    end
    return nothing
end

"""
    device(backend::Backend)::Int

Return the 1-based index of the currently active device for `backend`.

!!! note
    The default implementation assumes a single device. Backends supporting multiple devices
    **must** implement `device(backend::Backend)::Int`, [`ndevices`](@ref),
    and [`device!`](@ref).
"""
function device(::Backend)
    return 1
end

"""
    ndevices(backend::Backend)::Int

Return the number of devices available to `backend`.

!!! note
    The default implementation assumes a single device. Backends supporting multiple devices
    **must** implement `ndevices(backend::Backend)::Int`, [`device`](@ref),
    and [`device!`](@ref).
"""
function ndevices(::Backend)
    return 1
end

"""
    device!(backend::Backend, id::Int)::Nothing

Select the active device for `backend`. `id` is a 1-based device index and must satisfy
`1 <= id <= ndevices(backend)`.

# Example

```julia
device!(CUDABackend(), 2)  # use the second CUDA device
```

!!! note
    The default implementation assumes a single device. Backends supporting multiple devices
    **must** implement `device!(backend::Backend, id::Int)`, [`ndevices`](@ref),
    and [`device`](@ref).
"""
function device!(backend::Backend, id::Int)
    if !(0 < id <= ndevices(backend))
        throw(ArgumentError("Device id $id out of bounds."))
    end
    return nothing
end

"""
    pagelock!(::Backend, dest::AbstractArray)::Union{Nothing, Missing}

Pagelock (pin) a host memory buffer for a backend device. This may be necessary for [`copyto!`](@ref)
to perform asynchronously w.r.t to the host/

This function should return `nothing`; or `missing` if not implemented.


!!! note
    Backends **may** implement this function.
"""
function pagelock!(::Backend, x)
    return missing
end

"""
    unsafe_free!(x::AbstractArray)

Release the memory of an array for reuse by future allocations
and reduce pressure on the allocator.
After releasing the memory of an array, it should no longer be accessed.

!!! note
    On CPU backend this is always a no-op.

!!! note
    Backend implementations **may** implement this function.
    If not implemented for a particular backend, default action is a no-op.
    Otherwise, it should be defined for backend's array type.
"""
function unsafe_free! end

unsafe_free!(::AbstractArray) = return


"""
    supports_unified(::Backend)::Bool

Returns whether unified memory arrays are supported by the backend.

!!! note
    Backend implementations **should** implement this function
    only if they **do** support unified memory.
"""
supports_unified(::Backend) = false

"""
    supports_atomics(::Backend)::Bool

Returns whether `@atomic` operations are supported by the backend.

!!! note
    Backend implementations **must** implement this function
    only if they **do not** support atomic operations with Atomix.
"""
supports_atomics(::Backend) = true

"""
    supports_float64(::Backend)::Bool

Returns whether `Float64` values are supported by the backend.

!!! note
    Backend implementations **must** implement this function
    only if they **do not** support `Float64`.
"""
supports_float64(::Backend) = true

"""
    allocate(::Backend, Type, dims...; unified=false)::AbstractArray

Allocate a storage array appropriate for the computational backend. `unified=true`
allocates an array using unified memory if the backend supports it and throws otherwise.
Use [`supports_unified`](@ref) to determine whether it is supported by a backend.

!!! note
    Backend implementations **must** implement `allocate(::NewBackend, T, dims::Tuple)`
    Backend implementations **should** implement `allocate(::NewBackend, T, dims::Tuple; unified::Bool=false)`
"""
allocate(backend::Backend, T::Type, dims...; kwargs...) = allocate(backend, T, dims; kwargs...)
function allocate(backend::Backend, T::Type, dims::Tuple; unified::Union{Nothing, Bool} = nothing)
    if isnothing(unified)
        throw(MethodError(allocate, (backend, T, dims)))
    elseif unified
        throw(ArgumentError("`$(typeof(backend))` does not support unified memory. If you believe it does, please open a github issue."))
    else
        return allocate(backend, T, dims)
    end
end


"""
    zeros(::Backend, Type, dims...; unified=false)::AbstractArray

Allocate a storage array appropriate for the computational backend filled with zeros.
`unified=true` allocates an array using unified memory if the backend supports it and
throws otherwise.
"""
zeros(backend::Backend, T::Type, dims...; kwargs...) = zeros(backend, T, dims; kwargs...)
function zeros(backend::Backend, ::Type{T}, dims::Tuple; kwargs...) where {T}
    data = allocate(backend, T, dims...; kwargs...)
    fill!(data, zero(T))
    return data
end

"""
    ones(::Backend, Type, dims...; unified=false)::AbstractArray

Allocate a storage array appropriate for the computational backend filled with ones.
`unified=true` allocates an array using unified memory if the backend supports it and
throws otherwise.
"""
ones(backend::Backend, T::Type, dims...; kwargs...) = ones(backend, T, dims; kwargs...)
function ones(backend::Backend, ::Type{T}, dims::Tuple; kwargs...) where {T}
    data = allocate(backend, T, dims; kwargs...)
    fill!(data, one(T))
    return data
end


"""
    copyto!(::Backend, dest::AbstractArray, src::AbstractArray)

Perform an asynchronous `copyto!` operation that is execution ordered with respect to the back-end.

For most users, `Base.copyto!` should suffice, performance a simple, synchronous copy.
Only when you know you need asynchronicity w.r.t. the host, you should consider using
this asynchronous version, which requires additional lifetime guarantees as documented below.

!!! warning

    Because of the asynchronous nature of this operation, the user is required to guarantee that the lifetime
    of the source extends past the *completion* of the copy operation as to avoid a use-after-free. It is not
    sufficient to simply use `GC.@preserve` around the call to `copyto!`, because that only extends the
    lifetime past the operation getting queued. Instead, it may be required to `synchronize()`,
    or otherwise guarantee that the source will still be around when the copy is executed:

    ```julia
    arr = zeros(64)
    GC.@preserve arr begin
        copyto!(backend, arr, ...)
        # other operations
        synchronize(backend)
    end
    ```

!!! note

    On some back-ends it may be necessary to first call [`pagelock!`](@ref) on host memory
    to enable fully asynchronous behavior w.r.t to the host.

!!! note
    Backends **must** implement this function.
"""
function copyto! end

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
struct Kernel{B, Kern}
    backend::B
    kern::Kern
end

"""
    check_launch_args(numworkgroups, workgroupsize)

Validate the launch configuration passed to a [`Kernel`](@ref), throwing an
`ArgumentError` if either argument has more than 3 dimensions.

Backends may call this from their kernel-launch method instead of writing their
own check.
"""
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
