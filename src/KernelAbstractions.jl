module KernelAbstractions

export @kernel
export @Const, @localmem, @private, @uniform, @synchronize
export @index, @groupsize, @ndrange
export @print
export Backend, GPU, CPU
export synchronize, get_backend, allocate

import PrecompileTools

import Atomix: @atomic, @atomicswap, @atomicreplace

using MacroTools
using StaticArrays
using Adapt

"""
    @kernel function f(args) end

Takes a function definition and generates a [`Kernel`](@ref KernelAbstractions.Kernel) constructor from it.
The enclosed function is allowed to contain kernel language constructs.
In order to call it the kernel has first to be specialized on the backend
and then invoked on the arguments.

# Kernel language

- [`@Const`](@ref)
- [`@index`](@ref)
- [`@groupsize`](@ref)
- [`@ndrange`](@ref)
- [`@localmem`](@ref)
- [`@private`](@ref)
- [`@uniform`](@ref)
- [`@synchronize`](@ref)
- [`@print`](@ref)

# Kernel constructor

After defining a kernel function `f`, call `f(backend[, workgroupsize[, ndrange]])` to obtain a
[`Kernel`](@ref KernelAbstractions.Kernel) specialized for that backend. Workgroup size and `ndrange` can be fixed at
construction time (for fewer runtime checks and less recompilation) or supplied at launch:

```julia
f(backend)                    # dynamic workgroup size and ndrange
f(backend, 64)                # static workgroup size of 64
f(backend, 64, 1024)          # static workgroup size and ndrange
f(backend, 64, (128, 128))    # multi-dimensional ndrange
```

# Example

```julia
using KernelAbstractions

@kernel function vecadd(A, @Const(B))
    I = @index(Global)
    @inbounds A[I] += B[I]
end

dev = CPU()
A = ones(1024)
B = rand(1024)
vecadd(dev, 64)(A, B, ndrange=length(A))
synchronize(dev)
```
"""
macro kernel(expr)
    return __kernel(expr, #=force_inbounds=# false, #=unsafe_indices=# false)
end

"""
    @kernel config function f(args) end

This allows for two different configurations:

1. `cpu={true, false}`: Disables code-generation of the CPU function. This relaxes semantics such that KernelAbstractions primitives can be used in non-kernel functions.
2. `inbounds={false, true}`: Enables a forced `@inbounds` macro around the function definition in the case the user is using too many `@inbounds` already in their kernel. Note that this can lead to incorrect results, crashes, etc and is fundamentally unsafe. Be careful!
3. `unsafe_indices={false, true}`: Disables the implicit validation of indices, users must avoid `@index(Global)`.

- [`@context`](@ref)

!!! warning
    This is an experimental feature.

!!! note
    `cpu={true, false}` is deprecated for KernelAbstractions 1.0
"""
macro kernel(ex...)
    if length(ex) == 1
        return __kernel(ex[1], false, false)
    else
        unsafe_indices = false
        force_inbounds = false
        for i in 1:(length(ex) - 1)
            if ex[i] isa Expr && ex[i].head == :(=) &&
                    ex[i].args[1] == :cpu && ex[i].args[2] isa Bool
                #deprecated
            elseif ex[i] isa Expr && ex[i].head == :(=) &&
                    ex[i].args[1] == :inbounds && ex[i].args[2] isa Bool
                force_inbounds = ex[i].args[2]
            elseif ex[i] isa Expr && ex[i].head == :(=) &&
                    ex[i].args[1] == :unsafe_indices && ex[i].args[2] isa Bool
                unsafe_indices = ex[i].args[2]
            else
                error(
                    "Configuration should be of form:\n" *
                        "* `cpu=false`\n" *
                        "* `inbounds=true`\n" *
                        "* `unsafe_indices=true`\n" *
                        "got `", ex[i], "`",
                )
            end
        end
        return __kernel(ex[end], force_inbounds, unsafe_indices)
    end
end

"""
    @Const(A)

`@Const` is an argument annotiation that asserts that the memory reference
by `A` is both not written to as part of the kernel and that it does not alias
any other memory in the kernel.

!!! danger
    Violating those constraints will lead to arbitrary behaviour.

    As an example given a kernel signature `kernel(A, @Const(B))`, you are not
    allowed to call the kernel with `kernel(A, A)` or `kernel(A, view(A, :))`.
"""
macro Const end

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
    pagelock!(::Backend, dest::AbstractArray)

Pagelock (pin) a host memory buffer for a backend device. This may be necessary for [`copyto!`](@ref)
to perform asynchronously w.r.t to the host/

This function should return `nothing`; or `missing` if not implemented.


!!! note
    Backends **may** implement this function.
"""
function pagelock! end

"""
    synchronize(::Backend)

Synchronize the current backend.

!!! note
    Backend implementations **must** implement this function.
"""
function synchronize end

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
    Backend

Abstract supertype for all KernelAbstractions backends.

Concrete backends (for example `CUDABackend` from CUDA.jl or [`CPU`](@ref) from this package)
determine where arrays are allocated and where kernels execute. Use [`get_backend`](@ref) to
obtain the backend for an array and [`allocate`](@ref) to create storage on a backend.

# Example

```julia
backend = get_backend(A)
kernel = my_kernel(backend, 256)
kernel(A, ndrange=length(A))
synchronize(backend)
```
"""
abstract type Backend end

include("intrinsics.jl")
import .KernelIntrinsics as KI
export KernelIntrinsics

###
# Kernel language
# - @localmem
# - @private
# - @uniform
# - @synchronize
# - @index
# - @groupsize
# - @ndrange
###

"""
    groupsize(ctx)

Return the workgroup size as a tuple. Equivalent to [`@groupsize`](@ref) inside a kernel.
"""
function groupsize end

"""
    ndrange(ctx)

Return the launch `ndrange` as a tuple. Equivalent to [`@ndrange`](@ref) inside a kernel.
"""
function ndrange end

"""
    @groupsize()

Query the workgroupsize on the backend. This function returns
a tuple corresponding to kernel configuration. In order to get
the total size you can use `prod(@groupsize())`.
"""
macro groupsize()
    return quote
        $groupsize($(esc(:__ctx__)))
    end
end

"""
    @ndrange()

Query the ndrange on the backend. This function returns
a tuple corresponding to kernel configuration.
"""
macro ndrange()
    return quote
        $size($ndrange($(esc(:__ctx__))))
    end
end

"""
    @localmem T dims

Declare storage that is local to a workgroup.
"""
macro localmem(T, dims)
    # Stay in sync with CUDAnative
    id = gensym("static_shmem")

    return quote
        $SharedMemory($(esc(T)), Val($(esc(dims))), Val($(QuoteNode(id))))
    end
end

"""
    @private T dims

Declare storage that is local to each item in the workgroup. This can be safely used
across [`@synchronize`](@ref) statements. On a CPU, this will allocate additional implicit
dimensions to ensure correct localization.

For storage that only persists between `@synchronize` statements, an `MArray` can be used
instead.

See also [`@uniform`](@ref).

!!! note
    `@private` is deprecated for KernelAbstractions 1.0
"""
macro private(T, dims)
    if dims isa Integer
        dims = (dims,)
    end
    return quote
        $Scratchpad($(esc(:__ctx__)), $(esc(T)), Val($(esc(dims))))
    end
end

"""
    @private mem = 1

Creates a private local of `mem` per item in the workgroup. This can be safely used
across [`@synchronize`](@ref) statements.

!!! note
    `@private` is deprecated for KernelAbstractions 1.0
"""
macro private(expr)
    return esc(expr)
end

"""
    @uniform expr

`expr` is evaluated outside the workitem scope. This is useful for variable declarations
that span workitems, or are reused across `@synchronize` statements.

!!! note
    `@uniform` is deprecated for KernelAbstractions 1.0
"""
macro uniform(value)
    return esc(value)
end

"""
    @synchronize()

After a `@synchronize` statement all read and writes to global and local memory
from each thread in the workgroup are visible in from all other threads in the
workgroup.

!!! note
    `@synchronize()` must be encountered by all workitems of a work-group executing the kernel or by none at all.
"""
macro synchronize()
    return quote
        $__synchronize()
    end
end

"""
    @synchronize(cond)

After a `@synchronize` statement all read and writes to global and local memory
from each thread in the workgroup are visible in from all other threads in the
workgroup. `cond` is not allowed to have any visible sideffects.

# Platform differences
  - `GPU`: This synchronization will only occur if the `cond` evaluates.
  - `CPU`: This synchronization will always occur.

!!! warning
    This variant of the `@synchronize` macro violates the requirement that `@synchronize` must be encountered
    by all workitems of a work-group executing the kernel or by none at all.
    Since v`0.9.34` this version of the macro is deprecated and lowers to `@synchronize()`
"""
macro synchronize(cond)
    return quote
        $__synchronize()
    end
end

"""
    @context()

Access the hidden context object used by KernelAbstractions.

!!! warning
    Only valid to be used from a kernel with `cpu=false`.

!!! note
    `@context` will be supported on all backends in KernelAbstractions 1.0
```
function f(@context, a)
    I = @index(Global, Linear)
    a[I]
end

@kernel cpu=false function my_kernel(a)
    f(@context, a)
end
```
"""
macro context()
    return esc(:(__ctx__))
end

# Defined to keep cpu support for `__print`
@generated function KI._print(items...)
    str = ""
    args = []

    for i in 1:length(items)
        item = :(items[$i])
        T = items[i]
        if T <: Val
            item = QuoteNode(T.parameters[1])
        end
        push!(args, item)
    end

    return quote
        print($(args...))
    end
end

"""
    @print(items...)

This is a unified print statement.

# Platform differences
  - `GPU`: This will reorganize the items to print via `@cuprintf`
  - `CPU`: This will call `print(items...)`
"""
macro print(items...)

    args = Union{Val, Expr, Symbol}[]

    items = [items...]
    while true
        isempty(items) && break

        item = popfirst!(items)

        # handle string interpolation
        if isa(item, Expr) && item.head == :string
            items = vcat(item.args, items)
            continue
        end

        # expose literals to the generator by using Val types
        if isbits(item) # literal numbers, etc
            push!(args, Val(item))
        elseif isa(item, QuoteNode) # literal symbols
            push!(args, Val(item.value))
        elseif isa(item, String) # literal strings need to be interned
            push!(args, Val(Symbol(item)))
        else # actual values that will be passed to printf
            push!(args, item)
        end
    end

    return quote
        $__print($(map(esc, args)...))
    end
end

"""
    @index

The `@index` macro can be used to give you the index of a workitem within a kernel
function. It supports both the production of a linear index or a cartesian index.
A cartesian index is a general N-dimensional index that is derived from the iteration space.

# Index granularity

  - `Global`: Used to access global memory.
  - `Group`: The index of the `workgroup`.
  - `Local`: The within `workgroup` index.

# Index kind

  - `Linear`: Produces an `Int64` that can be used to linearly index into memory.
  - `Cartesian`: Produces a `CartesianIndex{N}` that can be used to index into memory.
  - `NTuple`: Produces a `NTuple{N}` that can be used to index into memory.

If the index kind is not provided it defaults to `Linear`, this is subject to change.

# Examples

```julia
@index(Global, Linear)
@index(Global, Cartesian)
@index(Local, Cartesian)
@index(Group, Linear)
@index(Local, NTuple)
@index(Global)
```
"""
macro index(locale, args...)
    if !(locale === :Global || locale === :Local || locale === :Group)
        error("@index requires as first argument either :Global, :Local or :Group")
    end

    if length(args) >= 1
        if args[1] === :Cartesian ||
                args[1] === :Linear ||
                args[1] === :NTuple
            indexkind = args[1]
            args = args[2:end]
        else
            indexkind = :Linear
        end
    else
        indexkind = :Linear
    end

    index_function = Symbol(:__index_, locale, :_, indexkind)
    return Expr(:call, GlobalRef(KernelAbstractions, index_function), esc(:__ctx__), map(esc, args)...)
end

###
# Internal kernel functions
###

@inline function __index_Local_Linear(ctx)
    return KI.get_local_id().x
end

@inline function __index_Group_Linear(ctx)
    return KI.get_group_id().x
end

@inline function __index_Global_Linear(ctx)
    return KI.get_global_id().x
end

@inline function __index_Local_Cartesian(ctx)
    return @inbounds workitems(__iterspace(ctx))[KI.get_local_id().x]
end
@inline function __index_Group_Cartesian(ctx)
    return @inbounds blocks(__iterspace(ctx))[KI.get_group_id().x]
end
@inline function __index_Global_Cartesian(ctx)
    return @inbounds expand(__iterspace(ctx), KI.get_group_id().x, KI.get_local_id().x)
end

@inline __index_Local_NTuple(ctx, I...) = Tuple(__index_Local_Cartesian(ctx, I...))
@inline __index_Group_NTuple(ctx, I...) = Tuple(__index_Group_Cartesian(ctx, I...))
@inline __index_Global_NTuple(ctx, I...) = Tuple(__index_Global_Cartesian(ctx, I...))

struct ConstAdaptor end

Adapt.adapt_storage(::ConstAdaptor, a::Array) = Base.Experimental.Const(a)

constify(arg) = adapt(ConstAdaptor(), arg)

###
# Backend hierarchy
###


"""
Abstract type for all GPU based KernelAbstractions backends.

!!! note
    New backend implementations **must** sub-type this abstract type.

!!! note
    `GPU` will be removed in KernelAbstractions v1.0
"""
abstract type GPU <: Backend end

"""
    get_backend(A::AbstractArray)::Backend

Get a [`Backend`](@ref) instance suitable for array `A`.

!!! note
    Backend implementations **must** provide `get_backend` for their custom array type.
    It should be the same as the return type of [`allocate`](@ref)
"""
function get_backend end

# Should cover SubArray, ReshapedArray, ReinterpretArray, Hermitian, AbstractTriangular, etc.:
function get_backend(A::AbstractArray)
    P = parent(A)
    if P isa typeof(A)
        throw(ArgumentError("Implement `KernelAbstractions.get_backend(::$(typeof(A)))`"))
    end
    return get_backend(P)
end

# Define:
#   adapt_storage(::Backend, a::Array) = adapt(BackendArray, a)
#   adapt_storage(::Backend, a::BackendArray) = a

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
    priority!(::Backend, prio::Symbol)

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
"""
function device(::Backend)
    return 1
end

"""
    ndevices(backend::Backend)::Int

Return the number of devices available to `backend`.
"""
function ndevices(::Backend)
    return 1
end

"""
    device!(backend::Backend, id::Int)

Select the active device for `backend`. `id` is a 1-based device index and must satisfy
`1 <= id <= ndevices(backend)`.

# Example

```julia
device!(CUDABackend(), 2)  # use the second CUDA device
```
"""
function device!(backend::Backend, id::Int)
    if !(0 < id <= ndevices(backend))
        throw(ArgumentError("Device id $id out of bounds."))
    end
    return nothing
end

"""
    functional(::Backend)

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

function pagelock!(::Backend, x)
    return missing
end

include("nditeration.jl")
using .NDIteration
import .NDIteration: get

###
# Kernel closure struct
###

"""
    Kernel{Backend, WorkgroupSize, NDRange, Func}

Host-side handle for a kernel specialized on a backend, workgroup size, and `ndrange`.

Kernels are created by calling a [`@kernel`](@ref) function on a backend, for example
`my_kernel(CUDABackend(), 256)`. The returned object is callable:

```julia
kernel = my_kernel(backend, 64)
kernel(A, B, ndrange=length(A))   # launch asynchronously
synchronize(backend)
```

Use [`workgroupsize`](@ref KernelAbstractions.workgroupsize), [`ndrange`](@ref KernelAbstractions.ndrange),
and [`backend`](@ref KernelAbstractions.backend) to inspect a kernel's static configuration.

!!! note
    Backend implementations **must** implement:
    ```
    (kernel::Kernel{<:NewBackend})(args...; ndrange=nothing, workgroupsize=nothing)
    ```
    As well as the on-device functionality.
"""
struct Kernel{Backend, WorkgroupSize <: _Size, NDRange <: _Size, Fun}
    backend::Backend
    f::Fun
end

function Base.similar(kernel::Kernel{D, WS, ND}, f::F) where {D, WS, ND, F}
    return Kernel{D, WS, ND, F}(kernel.backend, f)
end

"""
    workgroupsize(kernel::Kernel)

Return the static workgroup size type parameter of `kernel` (`StaticSize` or `DynamicSize`).
"""
function workgroupsize(::Kernel{D, WorkgroupSize}) where {D, WorkgroupSize}
    return WorkgroupSize
end

"""
    ndrange(kernel::Kernel)

Return the static `ndrange` type parameter of `kernel` (`StaticSize` or `DynamicSize`).
"""
function ndrange(::Kernel{D, WorkgroupSize, NDRange}) where {D, WorkgroupSize, NDRange}
    return NDRange
end

"""
    backend(kernel::Kernel)

Return the [`Backend`](@ref) that `kernel` was constructed for.
"""
function backend(kernel::Kernel)
    return kernel.backend
end

"""
    partition(kernel, ndrange, workgroupsize)

Partition the iteration space of `kernel` into workgroups.

Returns the blocked iteration space and whether dynamic bounds-checking is required for the
last (possibly partial) workgroup. Primarily used by backend implementations and tests.
"""
@inline function partition(kernel, ndrange, workgroupsize)
    static_ndrange = KernelAbstractions.ndrange(kernel)
    static_workgroupsize = KernelAbstractions.workgroupsize(kernel)

    if ndrange === nothing && static_ndrange <: DynamicSize ||
            workgroupsize === nothing && static_workgroupsize <: DynamicSize
        errmsg = """
            Can not partition kernel!

            You created a dynamically sized kernel, but forgot to provide runtime
            parameters for the kernel. Either provide them statically if known
            or dynamically.
            NDRange(Static):  $(static_ndrange)
            NDRange(Dynamic): $(ndrange)
            Workgroupsize(Static):  $(static_workgroupsize)
            Workgroupsize(Dynamic): $(workgroupsize)
        """
        error(errmsg)
    end

    if static_ndrange <: StaticSize
        if ndrange !== nothing && ndrange != get(static_ndrange)
            error("Static NDRange ($static_ndrange) and launch NDRange ($ndrange) differ")
        end
        ndrange = get(static_ndrange)
    end

    if static_workgroupsize <: StaticSize
        if workgroupsize !== nothing && workgroupsize != get(static_workgroupsize)
            error("Static WorkgroupSize ($static_workgroupsize) and launch WorkgroupSize $(workgroupsize) differ")
        end
        workgroupsize = get(static_workgroupsize)
    end

    @assert workgroupsize !== nothing
    @assert ndrange !== nothing
    blocks, workgroupsize, dynamic = NDIteration.partition(ndrange, workgroupsize)

    if static_ndrange <: StaticSize
        static_blocks = StaticSize{blocks}
        blocks = nothing
    else
        static_blocks = DynamicSize
        blocks = CartesianIndices(blocks)
    end

    if static_workgroupsize <: StaticSize
        static_workgroupsize = StaticSize{workgroupsize} # we might have padded workgroupsize
        workgroupsize = nothing
    else
        workgroupsize = CartesianIndices(workgroupsize)
    end

    iterspace = NDRange{length(ndrange), static_blocks, static_workgroupsize}(blocks, workgroupsize)
    return iterspace, dynamic
end

function construct(backend::Backend, ::S, ::NDRange, xpu_name::XPUName) where {Backend <: GPU, S <: _Size, NDRange <: _Size, XPUName}
    return Kernel{Backend, S, NDRange, XPUName}(backend, xpu_name)
end

###
# Compiler
###

include("compiler.jl")

###
# Compiler/Frontend
###

function __workitems_iterspace end
function __validindex end

# for reflection
function mkcontext end
function launch_config end

include("macros.jl")

###
# Backends/Interface
###

function Scratchpad end
SharedMemory(::Type{T}, dims::Val{Dims}, id::Val{Id}) where {T, Dims, Id} = KI.localmemory(T, dims)

__synchronize() = KI.barrier()

__print(args...) = KI._print(args...)

# Utils
__size(args::Tuple) = Tuple{args...}
__size(i::Int) = Tuple{i}

"""
    argconvert(kernel::Kernel, arg)

Convert `arg` to the device-side representation expected by `kernel`'s backend.

Backend implementations define methods for their array and scalar types. This is called
automatically when a kernel is launched.
"""
argconvert(k::Kernel{T}, arg) where {T} =
    error("Don't know how to convert arguments for Kernel{$T}")

# Enzyme support
supports_enzyme(::Backend) = false
function __fake_compiler_job end

###
# Extras
# - LoopInfo
###

include("extras/extras.jl")

include("reflection.jl")

# Initialized

@kernel function init_kernel(arr, f::F, ::Type{T}) where {F, T}
    I = @index(Global)
    @inbounds arr[I] = f(T)
end

@kernel function copy_kernel(A, @Const(B))
    I = @index(Global)
    @inbounds A[I] = B[I]
end

# CPU backend
include("pocl/pocl.jl")
using .POCL
export POCLBackend

"""
    POCLBackend()

CPU backend that compiles kernels to OpenCL via [POCL](https://portablecl.org/) and executes
them on the host. This is the concrete type behind the [`CPU`](@ref) alias.
"""
POCLBackend

"""
    CPU

Type alias for [`POCLBackend`](@ref), the CPU execution backend.

Construct with `CPU()` (equivalent to `POCLBackend()`). Kernels run on the host via POCL/OpenCL
using the same programming model as GPU backends, which is useful for debugging and for running
kernel code without a GPU.

# Example

```julia
A = ones(Float32, 1024)
mul2_kernel(CPU(), 64)(A, ndrange=length(A))
synchronize(CPU())
```
"""
const CPU = POCLBackend

# precompile
PrecompileTools.@compile_workload begin
    @eval begin
        @kernel function precompile_kernel(A, @Const(B))
            i = @index(Global, Linear)
            lmem = @localmem Float32 (5,)
            pmem = @private Float32 (1,)
            @synchronize
        end
    end
end

end #module
