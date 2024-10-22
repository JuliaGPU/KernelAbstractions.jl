module KernelAbstractions

export @kernel
export @Const, @localmem, @private, @uniform, @synchronize
export @index, @groupsize, @ndrange
export @print
export Backend, GPU, CPU
export synchronize, get_backend, allocate

import PrecompileTools

import Atomix: @atomic, @atomicswap, @atomicreplace
import UnsafeAtomics

using MacroTools
using StaticArrays
using Adapt

"""
    @kernel function f(args) end

Takes a function definition and generates a [`Kernel`](@ref) constructor from it.
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

# Example:

```julia
@kernel function vecadd(A, @Const(B))
    I = @index(Global)
    @inbounds A[I] += B[I]
end

A = ones(1024)
B = rand(1024)
vecadd(CPU(), 64)(A, B, ndrange=size(A))
synchronize(backend)
```
"""
macro kernel(expr)
    return __kernel(expr, #=generate_cpu=# true, #=force_inbounds=# false)
end

"""
    @kernel config function f(args) end

This allows for two different configurations:

1. `cpu={true, false}`: Disables code-generation of the CPU function. This relaxes semantics such that KernelAbstractions primitives can be used in non-kernel functions.
2. `inbounds={false, true}`: Enables a forced `@inbounds` macro around the function definition in the case the user is using too many `@inbounds` already in their kernel. Note that this can lead to incorrect results, crashes, etc and is fundamentally unsafe. Be careful!

- [`@context`](@ref)

!!! warn
    This is an experimental feature.
"""
macro kernel(ex...)
    if length(ex) == 1
        return __kernel(ex[1], true, false)
    else
        generate_cpu = true
        force_inbounds = false
        for i in 1:(length(ex) - 1)
            if ex[i] isa Expr && ex[i].head == :(=) &&
                    ex[i].args[1] == :cpu && ex[i].args[2] isa Bool
                generate_cpu = ex[i].args[2]
            elseif ex[i] isa Expr && ex[i].head == :(=) &&
                    ex[i].args[1] == :inbounds && ex[i].args[2] isa Bool
                force_inbounds = ex[i].args[2]
            else
                error(
                    "Configuration should be of form:\n" *
                        "* `cpu=true`\n" *
                        "* `inbounds=false`\n" *
                        "got `", ex[i], "`",
                )
            end
        end
        return __kernel(ex[end], generate_cpu, force_inbounds)
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

Perform a `copyto!` operation that execution ordered with respect to the backend.

!!! note
    Backend implementations **must** implement this function.
"""
function copyto! end

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

function groupsize end
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
"""
macro private(expr)
    return esc(expr)
end

"""
    @uniform expr

`expr` is evaluated outside the workitem scope. This is useful for variable declarations
that span workitems, or are reused across `@synchronize` statements.
"""
macro uniform(value)
    return esc(value)
end

"""
    @synchronize()

After a `@synchronize` statement all read and writes to global and local memory
from each thread in the workgroup are visible in from all other threads in the
workgroup.
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
"""
macro synchronize(cond)
    return quote
        $(esc(cond)) && $__synchronize()
    end
end

"""
    @context()

Access the hidden context object used by KernelAbstractions.

!!! warn
    Only valid to be used from a kernel with `cpu=false`.

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

function __index_Local_Linear end
function __index_Group_Linear end
function __index_Global_Linear end

function __index_Local_Cartesian end
function __index_Group_Cartesian end
function __index_Global_Cartesian end

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

Abstract type for all KernelAbstractions backends.
"""
abstract type Backend end

"""
Abstract type for all GPU based KernelAbstractions backends.

!!! note
    New backend implementations **must** sub-type this abstract type.
"""
abstract type GPU <: Backend end

"""
    CPU(; static=false)

Instantiate a CPU (multi-threaded) backend.

## Options:
 - `static`: Uses a static thread assignment, this can be beneficial for NUMA aware code.
   Defaults to false.
"""
struct CPU <: Backend
    static::Bool
    CPU(; static::Bool = false) = new(static)
end

"""
    isgpu(::Backend)::Bool

Returns true for all [`GPU`](@ref) backends.
"""
isgpu(::GPU) = true
isgpu(::CPU) = false


"""
    get_backend(A::AbstractArray)::Backend

Get a [`Backend`](@ref) instance suitable for array `A`.

!!! note
    Backend implementations **must** provide `get_backend` for their custom array type.
    It should be the same as the return type of [`allocate`](@ref)
"""
function get_backend end

# Should cover SubArray, ReshapedArray, ReinterpretArray, Hermitian, AbstractTriangular, etc.:
get_backend(A::AbstractArray) = get_backend(parent(A))

get_backend(::Array) = CPU()

# Define:
#   adapt_storage(::Backend, a::Array) = adapt(BackendArray, a)
#   adapt_storage(::Backend, a::BackendArray) = a
Adapt.adapt_storage(::CPU, a::Array) = a

"""
    allocate(::Backend, Type, dims...)::AbstractArray

Allocate a storage array appropriate for the computational backend.

!!! note
    Backend implementations **must** implement `allocate(::NewBackend, T, dims::Tuple)`
"""
allocate(backend::Backend, T, dims...) = allocate(backend, T, dims)
allocate(backend::Backend, T, dims::Tuple) = throw(MethodError(allocate, (backend, T, dims)))

"""
    zeros(::Backend, Type, dims...)::AbstractArray

Allocate a storage array appropriate for the computational backend filled with zeros.
"""
zeros(backend::Backend, T, dims...) = zeros(backend, T, dims)
function zeros(backend::Backend, ::Type{T}, dims::Tuple) where {T}
    data = allocate(backend, T, dims...)
    fill!(data, zero(T))
    return data
end

"""
    ones(::Backend, Type, dims...)::AbstractArray

Allocate a storage array appropriate for the computational backend filled with ones.
"""
ones(backend::Backend, T, dims...) = ones(backend, T, dims)
function ones(backend::Backend, ::Type{T}, dims::Tuple) where {T}
    data = allocate(backend, T, dims)
    fill!(data, one(T))
    return data
end

"""
    supports_atomics(::Backend)::Bool

Returns whether `@atomic` operations are supported by the backend.

!!! note
    Backend implementations **must** implement this function,
    only if they **do not** support atomic operations with Atomix.
"""
supports_atomics(::Backend) = true

"""
    supports_float64(::Backend)::Bool

Returns whether `Float64` values are supported by the backend.

!!! note
    Backend implementations **must** implement this function,
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

include("nditeration.jl")
using .NDIteration
import .NDIteration: get

###
# Kernel closure struct
###

"""
    Kernel{Backend, WorkgroupSize, NDRange, Func}

Kernel closure struct that is used to represent the backend
kernel on the host. `WorkgroupSize` is the number of workitems
in a workgroup.

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

workgroupsize(::Kernel{D, WorkgroupSize}) where {D, WorkgroupSize} = WorkgroupSize
ndrange(::Kernel{D, WorkgroupSize, NDRange}) where {D, WorkgroupSize, NDRange} = NDRange
backend(kernel::Kernel) = kernel.backend

"""
Partition a kernel for the given ndrange and workgroupsize.
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

function construct(backend::Backend, ::S, ::NDRange, xpu_name::XPUName) where {Backend <: Union{CPU, GPU}, S <: _Size, NDRange <: _Size, XPUName}
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

include("macros.jl")

###
# Backends/Interface
###

function Scratchpad end
function SharedMemory end

function __synchronize()
    error("@synchronize used outside kernel or not captured")
end

@generated function __print(items...)
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

# Utils
__size(args::Tuple) = Tuple{args...}
__size(i::Int) = Tuple{i}

"""
    argconvert(::Kernel, arg)

Convert arguments to the device side representation.
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

include("cpu.jl")

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

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869" include("../ext/EnzymeExt.jl")
    end
end

if !isdefined(Base, :get_extension)
    include("../ext/LinearAlgebraExt.jl")
    include("../ext/SparseArraysExt.jl")
end

end #module
