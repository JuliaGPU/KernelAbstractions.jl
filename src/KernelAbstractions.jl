module KernelAbstractions

export @kernel
export @Const, @localmem, @private, @synchronize, @index
export Device, GPU, CPU, CUDA 

using StaticArrays
using Cassette
using Requires

"""
   @kernel function f(args) end

Takes a function definition and generates a Kernel constructor from it.
The enclosed function is allowed to contain kernel language constructs.
In order to call it the kernel has first to be specialized on the backend
and then invoked on the arguments.

# Kernel language

- [`@Const`](@ref)
- [`@index`](@ref)
- [`@localmem`](@ref)
- [`@private`](@ref)
- [`@synchronize`](@ref)

# Example:

@kernel function vecadd(A, @Const(B))
    I = @index(Global)
    @inbounds A[I] += B[I]
end

A = ones(1024)
B = rand(1024)
event = vecadd(CPU(), 64)(A, B, ndrange=size(A))
wait(event)
"""
macro kernel(expr)
    __kernel(expr)
end

"""
   @Const(A)

`@Const` is an argument annotiation that asserts that the memory reference
by `A` is both not written to as part of the kernel and that it does not alias
any other memory in the kernel.

!!! danger
    Violating those constraints will lead to arbitrary behaviour.

as an example given a kernel signature `kernel(A, @Const(B))`, you are not
allowed to call the kernel with `kernel(A, A)` or `kernel(A, view(A, :))`.
"""
macro Const end

abstract type Event end
import Base.wait

# TODO
function async_copy! end
# function register end
# function unregister end

###
# Kernel language
# - @localmem
# - @private
# - @synchronize
# - @index
###

const shmem_id = Ref(0)

"""
   @localmem T dims
"""
macro localmem(T, dims)
    id = (shmem_id[]+= 1)

    quote
        $SharedMemory($(esc(T)), Val($(esc(dims))), Val($id))
    end
end

"""
   @private T dims
"""
macro private(T, dims)
    quote
        $Scratchpad($(esc(T)), Val($(esc(dims))))
    end
end

"""
   @synchronize()
"""
macro synchronize()
    @error "@synchronize not captured or used outside @kernel"
end

"""
   @index(Global)
   @index(Local)
   @index(Global, Cartesian)
"""
macro index(locale, args...)
    if !(locale === :Global || locale === :Local)
        error("@index requires as first argument either :Global or :Local")
    end

    if length(args) >= 1
        if args[1] === :Cartesian || args[1] === :Linear
            indexkind = args[1]
            args = args[2:end]
        else
            indexkind = :Linear
        end
    else
        indexkind = :Linear
    end

    if indexkind === :Cartesian && locale === :Local
        error("@index(Local, Cartesian) is not implemented yet") 
    end
    
    index_function = Symbol(:__index_, locale, :_, indexkind)
    Expr(:call, GlobalRef(KernelAbstractions, index_function), map(esc, args)...)
end

###
# Internal kernel functions
###

function __index_Local_Linear end
function __index_Global_Linear end

function __index_Local_Cartesian end
function __index_Global_Cartesian end
###
# Backend hierarchy
###

abstract type Device end
abstract type GPU <: Device end

struct CPU <: Device end
struct CUDA <: GPU end
# struct AMD <: GPU end
# struct Intel <: GPU end

###
# Kernel closure struct
###

import Base.@pure

abstract type _Size end
struct DynamicSize <: _Size end
struct StaticSize{S} <: _Size
    function StaticSize{S}() where S
        new{S::Tuple{Vararg{Int}}}()
    end
end

@pure StaticSize(s::Tuple{Vararg{Int}}) = StaticSize{s}() 
@pure StaticSize(s::Int...) = StaticSize{s}() 
@pure StaticSize(s::Type{<:Tuple}) = StaticSize{tuple(s.parameters...)}()

# Some @pure convenience functions for `StaticSize`
@pure get(::Type{StaticSize{S}}) where {S} = S
@pure get(::StaticSize{S}) where {S} = S
@pure Base.getindex(::StaticSize{S}, i::Int) where {S} = i <= length(S) ? S[i] : 1
@pure Base.ndims(::StaticSize{S}) where {S} = length(S)
@pure Base.length(::StaticSize{S}) where {S} = prod(S)

"""
    Kernel{Device, WorkgroupSize, NDRange, Func}

Kernel closure struct that is used to represent the device
kernel on the host. `WorkgroupSize` is the number of workitems
in a workgroup.
"""
struct Kernel{Device, WorkgroupSize<:_Size, NDRange<:_Size, Fun}
    f::Fun
end

workgroupsize(::Kernel{D, WorkgroupSize}) where {D, WorkgroupSize} = WorkgroupSize
ndrange(::Kernel{D, WorkgroupSize, NDRange}) where {D, WorkgroupSize,NDRange} = NDRange

"""
    partition(kernel, ndrange)

Splits the maximum size of the iteration space by the workgroupsize.
Returns the number of workgroups necessary and whether the last workgroup
needs to perform dynamic bounds-checking.
"""
@inline function partition(kernel::Kernel, ndrange, workgroupsize)
    static_ndrange = KernelAbstractions.ndrange(kernel)
    static_workgroupsize = KernelAbstractions.workgroupsize(kernel)

    if ndrange === nothing && static_ndrange <: DynamicSize ||
       workgroupsize === nothing && static_workgroupsize <: DynamicSize
        errmsg = """
            Can not partition kernel!

            You created a dynamically sized kernel, but forgot to provide runtime
            parameters for the kernel. Either provide them statically if known
            or dynamically.
            NDRange(Static):  $(typeof(static_ndrange))
            NDRange(Dynamic): $(ndrange)
            Workgroupsize(Static):  $(typeof(static_workgroupsize))
            Workgroupsize(Dynamic): $(workgroupsize)
        """
        error(errmsg)
    end

    if ndrange !== nothing && static_ndrange <: StaticSize
        if prod(ndrange) != prod(get(static_ndrange))
            error("Static NDRange and launch NDRange differ")
        end
    end

    if static_workgroupsize <: StaticSize
        @assert length(get(static_workgroupsize)) === 1
        static_workgroupsize = get(static_workgroupsize)[1]
        if workgroupsize !== nothing && workgroupsize != static_workgroupsize
            error("Static WorkgroupSize and launch WorkgroupSize differ")
        end
        workgroupsize = static_workgroupsize
    end
    @assert workgroupsize !== nothing

    if static_ndrange <: StaticSize
        maxsize = prod(get(static_ndrange))
    else
        maxsize = prod(ndrange)
    end

    nworkgroups = fld1(maxsize, workgroupsize)
    dynamic     = mod(maxsize, workgroupsize) != 0

    dynamic || @assert(nworkgroups * workgroupsize == maxsize)

    return nworkgroups, dynamic 
end

###
# Compiler/Cassette
###

include("compiler.jl")

###
# Compiler/Frontend
###

function groupsize end

@inline function __workitems_iterspace()
    return 1:groupsize()
end

function __validindex end

# TODO: GPU ConstWrapper that forwards loads to `ldg` and forbids stores
ConstWrapper(A) = A
include("macros.jl")

###
# Backends/Interface
###

function Scratchpad(::Type{T}, ::Val{Dims}) where {T, Dims}
    throw(MethodError(ScratchArray, (T, Val(Dims))))
end

function SharedMemory(::Type{T}, ::Val{Dims}, ::Val{Id}) where {T, Dims, Id}
    throw(MethodError(ScratchArray, (T, Val(Dims), Val(Id))))
end

###
# Backends/Implementation
###

# Utils
__size(args::Tuple) = Tuple{args...}
__size(i::Int) = Tuple{i}

include("backends/cpu.jl")
@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
    include("backends/cuda.jl")
end

###
# Extras
# - LoopInfo
###

include("extras/extras.jl")
end #module
