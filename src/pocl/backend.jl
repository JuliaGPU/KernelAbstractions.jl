module POCLKernels

using ..POCL
using ..POCL: @device_override, cl, method_table
using ..POCL: device, clconvert, clfunction

import KernelAbstractions as KA
import KernelAbstractions.KernelIntrinsics as KI

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
KA.synchronize(::POCLBackend) = cl.finish(cl.queue())
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

KI.kiconvert(::POCLBackend, arg) = clconvert(arg)

function KI.kifunction(::POCLBackend, f::F, tt::TT = Tuple{}; name = nothing, kwargs...) where {F, TT}
    kern = clfunction(f, tt; name, kwargs...)
    return KI.KIKernel{POCLBackend, typeof(kern)}(POCLBackend(), kern)
end

function (obj::KI.KIKernel{POCLBackend})(args...; numworkgroups = nothing, workgroupsize = nothing)
    local_size = StaticArrays.MVector{3}((1, 1, 1))
    if !isnothing(workgroupsize)
        for (i, val) in enumerate(workgroupsize)
            local_size[i] = val
        end
    end

    global_size = StaticArrays.MVector{3}((1, 1, 1))
    if !isnothing(numworkgroups)
        for (i, val) in enumerate(numworkgroups)
            global_size[i] = val * local_size[i]
        end
    end

    return obj.kern(args...; local_size, global_size)
end

function KI.kernel_max_work_group_size(::POCLBackend, kikern::KI.KIKernel{<:POCLBackend}; max_work_items::Int = typemax(Int))::Int
    wginfo = cl.work_group_info(kikern.kern.fun, device())
    return Int(min(wginfo.size, max_work_items))
end
function KI.max_work_group_size(::POCLBackend)::Int
    return Int(device().max_work_group_size)
end
function KI.multiprocessor_count(::POCLBackend)::Int
    return Int(device().max_compute_units)
end

## Indexing Functions

@device_override @inline function KI.get_local_id()
    return (; x = Int(get_local_id(1)), y = Int(get_local_id(2)), z = Int(get_local_id(3)))
end

@device_override @inline function KI.get_group_id()
    return (; x = Int(get_group_id(1)), y = Int(get_group_id(2)), z = Int(get_group_id(3)))
end

@device_override @inline function KI.get_global_id()
    return (; x = Int(get_global_id(1)), y = Int(get_global_id(2)), z = Int(get_global_id(3)))
end

@device_override @inline function KI.get_local_size()
    return (; x = Int(get_local_size(1)), y = Int(get_local_size(2)), z = Int(get_local_size(3)))
end

@device_override @inline function KI.get_num_groups()
    return (; x = Int(get_num_groups(1)), y = Int(get_num_groups(2)), z = Int(get_num_groups(3)))
end

@device_override @inline function KI.get_global_size()
    return (; x = Int(get_global_size(1)), y = Int(get_global_size(2)), z = Int(get_global_size(3)))
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

@device_override @inline function KI.localmemory(::Type{T}, ::Val{Dims}) where {T, Dims}
    ptr = POCL.emit_localmemory(T, Val(prod(Dims)))
    CLDeviceArray(Dims, ptr)
end

@device_override @inline function KA.Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    StaticArrays.MArray{KA.__size(Dims), T}(undef)
end


## Synchronization and Printing

@device_override @inline function KI.barrier()
    work_group_barrier(POCL.LOCAL_MEM_FENCE | POCL.GLOBAL_MEM_FENCE)
end

@device_override @inline function KI._print(args...)
    POCL._print(args...)
end


## Other

KA.argconvert(::KA.Kernel{POCLBackend}, arg) = clconvert(arg)

end
