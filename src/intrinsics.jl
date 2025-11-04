module KernelIntrinsics

import ..KernelAbstractions: Backend
import GPUCompiler: split_kwargs, assign_args!

"""
    get_global_size()::@NamedTuple{x::Int, y::Int, z::Int}

Return the number of global work-items specified.
"""
function get_global_size end

"""
    get_global_id()::@NamedTuple{x::Int, y::Int, z::Int}

Returns the unique global work-item ID.

!!! note
    1-based.
"""
function get_global_id end

"""
    get_local_size()::@NamedTuple{x::Int, y::Int, z::Int}

Return the number of local work-items specified.
"""
function get_local_size end

"""
    get_local_id()::@NamedTuple{x::Int, y::Int, z::Int}

Returns the unique local work-item ID.

!!! note
    1-based.
"""
function get_local_id end

"""
    get_num_groups()::@NamedTuple{x::Int, y::Int, z::Int}

Returns the number of groups.
"""
function get_num_groups end

"""
    get_group_id()::@NamedTuple{x::Int, y::Int, z::Int}

Returns the unique group ID.

!!! note
    1-based.
"""
function get_group_id end

"""
    localmemory(T, dims)

Declare memory that is local to a workgroup.

!!! note
    Backend implementations **must** implement:
    ```
    localmemory(T::DataType, ::Val{Dims}) where {T, Dims}
    ```
    As well as the on-device functionality.
"""
localmemory(::Type{T}, dims) where T = localmemory(T, Val(dims))
# @inline localmemory(::Type{T}, dims::Val{Dims}) where {T, Dims} = localmemory(T, dims, Val(gensym("static_shmem")))

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
    As well as the on-device functionality.
"""
function barrier()
    error("Group barrier used outside kernel or not captured")
end

# TODO
function print end


"""
    KIKernel{Backend, BKern}

KIKernel closure struct that is used to represent the backend
kernel on the host.

!!! note
    Backend implementations **must** implement:
    ```
    (kernel::KIKernel{<:NewBackend})(args...; numworkgroups=nothing, workgroupsize=nothing, kwargs...)
    ```
    As well as the on-device functionality.
"""
struct KIKernel{B, Kern}
    backend::B
    kern::Kern
end

"""
    kernel_max_work_group_size(backend, kern; [max_work_items::Int])::Int

The maximum workgroup size limit for a kernel as reported by the backend.
This function should always be used to determine the workgroup size before
launching a kernel.

!!! note
    Backend implementations **must** implement:
    ```
    kernel_max_work_group_size(backend::NewBackend, kern::KIKernel{<:NewBackend}; max_work_items::Int=typemax(Int))::Int
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

# TODO: docstring
# kiconvert(::NewBackend, arg)
function kiconvert end

# TODO: docstring
# KI.kifunction(::NewBackend, f::F, tt::TT=Tuple{}; name=nothing, kwargs...) where {F,TT}
function kifunction end

const MACRO_KWARGS = [:launch, :backend]
const COMPILER_KWARGS = [:name]
const LAUNCH_KWARGS = [:numworkgroups, :workgroupsize]

macro kikernel(backend, ex...)
    call = ex[end]
    kwargs = map(ex[1:end-1]) do kwarg
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
        key,val = first(other_kwargs).args
        throw(ArgumentError("Unsupported keyword argument '$key'"))
    end

    # handle keyword arguments that influence the macro's behavior
    launch = true
    for kwarg in macro_kwargs
        key,val = kwarg.args
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
    push!(code.args,
        quote
            $f_var = $f
            GC.@preserve $(vars...) $f_var begin
                $kernel_f = $kiconvert($backend, $f_var)
                $kernel_args = map(x -> $kiconvert($backend, x), ($(var_exprs...),))
                $kernel_tt = Tuple{map(Core.Typeof, $kernel_args)...}
                $kernel = $kifunction($backend, $kernel_f, $kernel_tt; $(compiler_kwargs...))
                if $launch
                    $kernel($(var_exprs...); $(call_kwargs...))
                end
                $kernel
            end
         end)

    return esc(quote
        let
            $code
        end
    end)
end

end
