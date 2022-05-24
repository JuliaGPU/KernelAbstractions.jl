module OneKernels

import oneAPI
import StaticArrays
import StaticArrays: MArray
import Adapt
import KernelAbstractions

export OneDevice

KernelAbstractions.get_device(::oneAPI.oneArray) = OneDevice()

import KernelAbstractions: Event, CPUEvent, NoneEvent, MultiEvent, CPU, GPU, isdone, failed

struct OneDevice <: GPU end

struct OneEvent <: Event end

import Base: wait

wait(ev::OneEvent) = oneAPI.oneL0.synchronize()

###
# async_copy
###

function KernelAbstractions.async_copy!(::OneDevice, A, B; dependencies=nothing, progress=yield)
    A isa Array && __pin!(A)
    B isa Array && __pin!(B)

    wait(OneEvent())
    GC.@preserve A B begin
        destptr = pointer(A)
        srcptr  = pointer(B)
        N       = length(A)
        oneAPI.unsafe_copyto!(oneAPI.context(), oneAPI.device(), destptr, srcptr, N)
    end

    return OneEvent()
end

import KernelAbstractions: Kernel, StaticSize, DynamicSize, partition, blocks, workitems, launch_config

###
# Kernel launch
###
function launch_config(kernel::Kernel{OneDevice}, ndrange, workgroupsize)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize, )
    end

    # partition checked that the ndrange's agreed
    if KernelAbstractions.ndrange(kernel) <: StaticSize
        ndrange = nothing
    end

    iterspace, dynamic = if KernelAbstractions.workgroupsize(kernel) <: DynamicSize &&
        workgroupsize === nothing
        # use ndrange as preliminary workgroupsize for autotuning
        partition(kernel, ndrange, ndrange)
    else
        partition(kernel, ndrange, workgroupsize)
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

function (obj::Kernel{OneDevice})(args...; ndrange=nothing, dependencies=Event(OneDevice()), workgroupsize=nothing, progress=yield)

    ndrange, workgroupsize, iterspace, dynamic = launch_config(obj, ndrange, workgroupsize)
    # this might not be the final context, since we may tune the workgroupsize
    ctx = mkcontext(obj, ndrange, iterspace)

    nblocks = length(blocks(iterspace))
    threads = length(workitems(iterspace))

    wait(OneEvent())

    # Launch kernel
    oneAPI.@oneapi(items=(threads,nblocks), groups=nblocks, obj.f(ctx, args...)) 
    return OneEvent()
end

import oneAPI: @device_override

import KernelAbstractions: CompilerMetadata, DynamicCheck, LinearIndices
import KernelAbstractions: __index_Local_Linear, __index_Group_Linear, __index_Global_Linear, __index_Local_Cartesian, __index_Group_Cartesian, __index_Global_Cartesian, __validindex, __print
import KernelAbstractions: mkcontext, expand, __iterspace, __ndrange, __dynamic_checkbounds

function mkcontext(kernel::Kernel{OneDevice}, _ndrange, iterspace)
    CompilerMetadata{KernelAbstractions.ndrange(kernel), DynamicCheck}(_ndrange, iterspace)
end

@device_override @inline function __index_Local_Linear(ctx)
    return oneAPI.get_local_linear_id()
end

@device_override @inline function __index_Group_Linear(ctx)
    return oneAPI.get_group_id()
end

@device_override @inline function __index_Global_Linear(ctx)
    return oneAPI.get_global_linear_id()
end

@device_override @inline function __validindex(ctx)
    if __dynamic_checkbounds(ctx)
        I = @inbounds expand(__iterspace(ctx), oneAPI.get_group_id(), oneAPI.get_local_linear_id())
        return I in __ndrange(ctx)
    else
        return true
    end
end

import KernelAbstractions: groupsize, __groupsize, __workitems_iterspace, add_float_contract, sub_float_contract, mul_float_contract

import KernelAbstractions: ConstAdaptor, SharedMemory, Scratchpad, __synchronize, __size

@device_override @inline function __synchronize()
    oneAPI.barrier()
end

@device_override @inline function __print(args...)
    oneAPI.@print(args...)
end

# Argument conversion
KernelAbstractions.argconvert(::Kernel{OneDevice}, arg) = oneAPI.adapt(oneAPI.Adaptor(), arg)

end
