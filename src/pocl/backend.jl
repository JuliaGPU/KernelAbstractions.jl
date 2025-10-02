module POCLKernels

using ..POCL
using ..POCL: @device_override, cl, method_table
using ..POCL: device

import KernelAbstractions as KA

import StaticArrays

import Adapt


## Back-end Definition

export POCLBackend

struct POCLBackend <: KA.GPU
end


## Memory Operations

KA.allocate(::POCLBackend, ::Type{T}, dims::Tuple; unified::Bool = false) where {T} = Array{T}(undef, dims)

function KA.zeros(backend::POCLBackend, ::Type{T}, dims::Tuple; kwargs...) where {T}
    arr = KA.allocate(backend, T, dims; kwargs...)
    kernel = KA.init_kernel(backend)
    kernel(arr, zero, T, ndrange = length(arr))
    return arr
end
function KA.ones(backend::POCLBackend, ::Type{T}, dims::Tuple; kwargs...) where {T}
    arr = KA.allocate(backend, T, dims; kwargs...)
    kernel = KA.init_kernel(backend)
    kernel(arr, one, T; ndrange = length(arr))
    return arr
end

function KA.copyto!(backend::POCLBackend, A, B)
    if KA.get_backend(A) == KA.get_backend(B) && KA.get_backend(A) isa POCLBackend
        if length(A) != length(B)
            error("Arrays must match in length")
        end
        if Base.mightalias(A, B)
            error("Arrays may not alias")
        end
        kernel = KA.copy_kernel(backend)
        kernel(A, B, ndrange = length(A))
        return A
    else
        return Base.copyto!(A, B)
    end
end

KA.functional(::POCLBackend) = true
KA.pagelock!(::POCLBackend, x) = nothing

KA.get_backend(::Array) = POCLBackend()
KA.synchronize(::POCLBackend) = nothing
KA.supports_float64(::POCLBackend) = true
KA.supports_unified(::POCLBackend) = true


## Kernel Launch

function KA.mkcontext(kernel::KA.Kernel{POCLBackend}, _ndrange, iterspace)
    return KA.CompilerMetadata{KA.ndrange(kernel), KA.DynamicCheck}(_ndrange, iterspace)
end
function KA.mkcontext(
        kernel::KA.Kernel{POCLBackend}, I, _ndrange, iterspace,
        ::Dynamic
    ) where {Dynamic}
    return KA.CompilerMetadata{KA.ndrange(kernel), Dynamic}(I, _ndrange, iterspace)
end

function KA.launch_config(kernel::KA.Kernel{POCLBackend}, ndrange, workgroupsize)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize,)
    end

    # partition checked that the ndrange's agreed
    if KA.ndrange(kernel) <: KA.StaticSize
        ndrange = nothing
    end

    iterspace, dynamic = if KA.workgroupsize(kernel) <: KA.DynamicSize &&
            workgroupsize === nothing
        # use ndrange as preliminary workgroupsize for autotuning
        KA.partition(kernel, ndrange, ndrange)
    else
        KA.partition(kernel, ndrange, workgroupsize)
    end

    return ndrange, workgroupsize, iterspace, dynamic
end

function threads_to_workgroupsize(threads, ndrange)
    total = 1
    return map(ndrange) do n
        x = min(div(threads, total), n)
        total *= x
        return x
    end
end

function (obj::KA.Kernel{POCLBackend})(args...; ndrange = nothing, workgroupsize = nothing)
    ndrange, workgroupsize, iterspace, dynamic =
        KA.launch_config(obj, ndrange, workgroupsize)

    # this might not be the final context, since we may tune the workgroupsize
    ctx = KA.mkcontext(obj, ndrange, iterspace)
    kernel = @opencl launch = false obj.f(ctx, args...)

    # figure out the optimal workgroupsize automatically
    if KA.workgroupsize(obj) <: KA.DynamicSize && workgroupsize === nothing
        wg_info = cl.work_group_info(kernel.fun, device())
        wg_size_nd = threads_to_workgroupsize(wg_info.size, ndrange)
        iterspace, dynamic = KA.partition(obj, ndrange, wg_size_nd)
        ctx = KA.mkcontext(obj, ndrange, iterspace)
    end

    groups = length(KA.blocks(iterspace))
    items = length(KA.workitems(iterspace))

    if groups == 0
        return nothing
    end

    # Launch kernel
    global_size = groups * items
    local_size = items
    event = kernel(ctx, args...; global_size, local_size)
    wait(event)
    cl.clReleaseEvent(event)
    return nothing
end


## Indexing Functions

@device_override @inline function KA.__index_Local_Linear(ctx)
    return get_local_id(1)
end

@device_override @inline function KA.__index_Group_Linear(ctx)
    return get_group_id(1)
end

@device_override @inline function KA.__index_Global_Linear(ctx)
    return get_global_id(1)
end

@device_override @inline function KA.__index_Local_Cartesian(ctx)
    @inbounds KA.workitems(KA.__iterspace(ctx))[get_local_id(1)]
end

@device_override @inline function KA.__index_Group_Cartesian(ctx)
    @inbounds KA.blocks(KA.__iterspace(ctx))[get_group_id(1)]
end

@device_override @inline function KA.__index_Global_Cartesian(ctx)
    return @inbounds KA.expand(KA.__iterspace(ctx), get_group_id(1), get_local_id(1))
end

@device_override @inline function KA.__validindex(ctx)
    if KA.__dynamic_checkbounds(ctx)
        I = @inbounds KA.expand(KA.__iterspace(ctx), get_group_id(1), get_local_id(1))
        return I in KA.__ndrange(ctx)
    else
        return true
    end
end


## Shared and Scratch Memory

@device_override @inline function KA.SharedMemory(::Type{T}, ::Val{Dims}, ::Val{Id}) where {T, Dims, Id}
    ptr = POCL.emit_localmemory(T, Val(prod(Dims)))
    CLDeviceArray(Dims, ptr)
end

@device_override @inline function KA.Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    StaticArrays.MArray{KA.__size(Dims), T}(undef)
end


## Synchronization and Printing

@device_override @inline function KA.__synchronize()
    work_group_barrier.(POCL.LOCAL_MEM_FENCE | POCL.GLOBAL_MEM_FENCE)
end

@device_override @inline function KA.__print(args...)
    POCL._print(args...)
end


## Other

KA.argconvert(::KA.Kernel{POCLBackend}, arg) = clconvert(arg)

end
