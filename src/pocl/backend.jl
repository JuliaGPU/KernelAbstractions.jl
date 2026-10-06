module POCLKernels

using ..POCL
using ..POCL: @device_override, cl, method_table
using ..POCL: device, device_limits, clconvert, clfunction

using SPIRV_LLVM_Backend_jll, SPIRV_Tools_jll

import KernelAbstractions as KA
import KernelInterface as KI

import SPIRVIntrinsics

import Adapt


## Back-end Definition

export POCLBackend

struct POCLBackend <: KI.Backend
end

function KI.versioninfo(io::IO, ::POCLBackend)
    println(io, "KernelAbstractions.jl version $(pkgversion(@__MODULE__))")
    println(io)

    println(io, "Toolchain:")
    println(io, " - Julia v$(VERSION)")
    for jll in [SPIRV_LLVM_Backend_jll, SPIRV_Tools_jll, cl.pocl_standalone_jll]
        name = string(jll)
        println(io, " - $(name[1:(end - 4)]): $(pkgversion(jll))")
    end
    println(io)

    println(io, "Julia packages:")
    for name in [:GPUCompiler, :LLVM, :KernelInterface, :SPIRVIntrinsics]
        mod = getfield(POCL, name)
        println(io, "- $(name): $(Base.pkgversion(mod))")
    end
    println(io)

    println(io, "POCL Version: ")
    for platform in cl.platforms()
        print(io, "  OpenCL $(platform.opencl_version.major).$(platform.opencl_version.minor)")
        if !isempty(platform.version)
            print(io, ", $(platform.version)")
        end
        println(io)

        for device in cl.devices(platform)
            print(io, "  · $(device.name)")

            # show a list of tags
            tags = []
            ## relevant extensions
            if in("cl_khr_fp16", device.extensions)
                push!(tags, "fp16")
            end
            if in("cl_khr_fp64", device.extensions)
                push!(tags, "fp64")
            end
            if in("cl_khr_il_program", device.extensions)
                push!(tags, "il")
            end
            ## render
            if !isempty(tags)
                print(io, " (", join(tags, ", "), ")")
            end
            println(io)
        end
    end
    return
end

## Memory Operations

# GPUCompiler performs 8- and 16-bit atomics on the aligned 32-bit word containing the value,
# which must not overlap another allocation or storage that is modified independently. That
# includes adjacent elements of the same array: while an 8- or 16-bit element is updated
# atomically, the others in its word must not be modified by plain stores concurrently. For
# arrays of a bits type backed by Julia-owned storage (from here, but also from `resize!`,
# `copy`, `similar`, ...), that word stays within the same allocation by how Julia's
# allocator is implemented, not by anything it documents:
# - Julia 1.11+ (v1.13.1 `src/genericmemory.c`, `jl_alloc_genericmemory_unchecked`): small
#   `Memory` data starts 16 bytes into an object from a GC pool, whose size classes
#   (`jl_gc_sizeclasses` in `src/julia_internal.h`) are multiples of 8 bytes, with objects
#   16-byte aligned (`GC_PAGE_OFFSET` in `src/gc-stock.h`); larger data comes from
#   `jl_gc_managed_malloc` (`src/gc-stock.c`), which rounds the size up to, and aligns to,
#   `JL_CACHE_BYTE_ALIGNMENT` (64 or 128 bytes). With MMTk (`src/gc-mmtk.c`), objects are
#   rounded up to their 16-byte alignment (`jl_mmtk_gc_alloc_default`) and
#   `jl_gc_managed_malloc` rounds the same way.
# - Julia 1.10 (v1.10.10 `src/array.c`, `_new_array_`): small arrays store their data at
#   least 8-byte aligned after the header, in a pool object of the same size classes; larger
#   ones use `jl_gc_managed_malloc` and `gc_managed_realloc_` (`src/gc.c`), which round and
#   align as above.
# Revisit this when Julia's allocator changes. Arrays wrapping foreign memory (e.g., with
# `unsafe_wrap`) are the user's responsibility, as documented for `CPU`.
KI.allocate(::POCLBackend, ::Type{T}, dims::Tuple; unified::Bool = false) where {T} = Array{T}(undef, dims)

#  Adapt.jl's `Array` rule converts every `AbstractArray` leaf; `isbits` arrays (ranges, view indices)
# which we want to keep as they are.
Adapt.adapt_storage(::POCLBackend, x::AbstractArray) = isbits(x) ? x : Adapt.adapt(Array, x)

# `@Const` applies `constify` inside the kernel, where arguments have already been
# converted to device arrays, so the rule has to be registered for `CLDeviceArray`
# rather than for `Array`.
Adapt.adapt_storage(::KA.ConstAdaptor, a::POCL.CLDeviceArray) = Base.Experimental.Const(a)


# Copying

KA.@kernel function copy_kernel(A, @Const(B))
    I = KA.@index(Global)
    @inbounds A[I] = B[I]
end


function KI.copyto!(backend::POCLBackend, A, B)
    length(A) == length(B) ||
        throw(ArgumentError("Arrays must match in length, got $(length(A)) and $(length(B))"))
    if KI.get_backend(A) == KI.get_backend(B) && KI.get_backend(A) isa POCLBackend
        if Base.mightalias(A, B)
            error("Arrays may not alias")
        end
        kernel = copy_kernel(backend)
        kernel(A, B, ndrange = length(A))
        return A
    else
        Base.copyto!(A, B)
        return A
    end
end

KI.functional(::POCLBackend) = true
KA.pagelock!(::POCLBackend, x) = nothing

KI.get_backend(::Array) = POCLBackend()

## Implementation note:
## The POCL backend uses `Base.Array` as it's array type, so the external operations
## `broadcast`, `*` and other high-level operations are handled by Julia. In order
## to provide the same memory synchronization semantics as other backends, we
## must synchronize upon kernel launch and can't rely on synchronization upon
## array access. Therefore, `synchronize` is a no-op.
KI.synchronize(::POCLBackend) = nothing
KI.supports_float64(::POCLBackend) = "cl_khr_fp64" in device().extensions
KI.supports_unified(::POCLBackend) = true
KI.supports_atomics(::POCLBackend) = true


## Kernel Launch

KI.argconvert(::POCLBackend, arg) = clconvert(arg)

# a compiled kernel, and the callable it was compiled from. the compiled kernel only holds
# pointers to the arrays the callable captures, so the callable has to be kept alive.
struct POCLKernel{K, F}
    kernel::K
    f::F
end

function KI.kernel_function(backend::POCLBackend, f::F, tt::TT = Tuple{}; name = nothing, kwargs...) where {F, TT}
    # fix the sub-group width, as `KI.sub_group_size` promises. pass it even if the device
    # has no sub-groups, so that `clfunction` is only compiled for one set of keywords.
    sub_group_size = device_limits().sub_group_size
    sub_group_size = sub_group_size > 0 ? sub_group_size : nothing
    kernel = clfunction(clconvert(f), tt; name, sub_group_size, kwargs...)
    kern = POCLKernel(kernel, f)
    return KI.Kernel{POCLBackend, typeof(kern)}(backend, kern)
end

function KI.launch(obj::KI.Kernel{POCLBackend}, groups::Dims{3}, items::Dims{3}, args::Tuple)
    # POCL launches synchronously, see the implementation note on `synchronize`. the
    # compiled kernel only holds pointers to the arrays captured by `f`, so keep it alive
    # until the kernel completes.
    f = obj.kern.f
    GC.@preserve f POCL.launch_and_wait(
        obj.kern.kernel, args; local_size = items, global_size = groups .* items
    )
    return nothing
end

function KI.max_work_group_size(kernel::KI.Kernel{<:POCLBackend})::Int
    wginfo = cl.work_group_info(kernel.kern.kernel.fun, device())
    return Int(wginfo.size)
end
KI.max_work_group_size(::POCLBackend)::Int = device_limits().max_work_group_size
KI.max_work_group_dims(::POCLBackend)::NTuple{3, Int} = device_limits().max_work_group_dims
# the grid is only limited by the size of `size_t`
KI.max_num_groups(::POCLBackend)::NTuple{3, Int} = (typemax(Int), typemax(Int), typemax(Int))
KI.sub_group_size(::POCLBackend)::Int = device_limits().sub_group_size
function KI.multiprocessor_count(::POCLBackend)::Int
    return Int(device().max_compute_units)
end

KI.supports_subgroups(::POCLBackend) = device_limits().sub_group_size > 0
# the types `sub_group_shuffle` supports; other types are shuffled field by field
const ShuffleTypes = Union{SPIRVIntrinsics.gentypes...}

function KI.supports_shuffle(backend::POCLBackend, ::Type{T}) where {T <: ShuffleTypes}
    KI.supports_subgroups(backend) || return false
    T === Float64 && return "cl_khr_fp64" in device().extensions
    T === Float16 && return "cl_khr_fp16" in device().extensions
    return true
end

## Indexing Functions

# `% T` rather than `T(x)`: a checked conversion leaves an error branch in every kernel.
# This needs SPIRVIntrinsics 1.1.3, whose 3-D builtins survive the truncation.

@device_override @inline function KI.get_local_id(::Type{T}) where {T}
    return (; x = get_local_id(1) % T, y = get_local_id(2) % T, z = get_local_id(3) % T)
end

@device_override @inline function KI.get_group_id(::Type{T}) where {T}
    return (; x = get_group_id(1) % T, y = get_group_id(2) % T, z = get_group_id(3) % T)
end

@device_override @inline function KI.get_local_size(::Type{T}) where {T}
    return (; x = get_local_size(1) % T, y = get_local_size(2) % T, z = get_local_size(3) % T)
end

@device_override @inline function KI.get_num_groups(::Type{T}) where {T}
    return (; x = get_num_groups(1) % T, y = get_num_groups(2) % T, z = get_num_groups(3) % T)
end

@device_override @inline function KI.get_global_id(::Type{T}) where {T}
    return (; x = get_global_id(1) % T, y = get_global_id(2) % T, z = get_global_id(3) % T)
end

@device_override @inline function KI.get_global_size(::Type{T}) where {T}
    return (; x = get_global_size(1) % T, y = get_global_size(2) % T, z = get_global_size(3) % T)
end

@device_override KI.get_sub_group_size(::Type{T}) where {T} = get_sub_group_size() % T

@device_override KI.get_max_sub_group_size(::Type{T}) where {T} = get_max_sub_group_size() % T

@device_override KI.get_num_sub_groups(::Type{T}) where {T} = get_num_sub_groups() % T

@device_override KI.get_sub_group_id(::Type{T}) where {T} = get_sub_group_id() % T

@device_override KI.get_sub_group_local_id(::Type{T}) where {T} = get_sub_group_local_id() % T

## Shared Memory

@device_override @inline function KI.localmemory(::Type{T}, ::Val{Dims}) where {T, Dims}
    ptr = POCL.emit_localmemory(T, Val(prod(Dims)))
    CLDeviceArray(Dims, ptr)
end


## Synchronization and Printing

@device_override @inline function KI.barrier()
    work_group_barrier(POCL.LOCAL_MEM_FENCE | POCL.GLOBAL_MEM_FENCE)
end

@device_override @inline function KI.sub_group_barrier()
    sub_group_barrier(POCL.LOCAL_MEM_FENCE | POCL.GLOBAL_MEM_FENCE)
end

@device_override KI.shfl(val::T, lane::Integer) where {T <: ShuffleTypes} =
    sub_group_shuffle(val, lane)

# past the sub-group width, `shfl_down` and `shfl_up` return the work-item's own value, which
# `sub_group_shuffle` (like SPIR-V's `OpGroupNonUniformShuffleDown`) leaves undefined
@device_override function KI.shfl_down(val::T, offset::Integer) where {T <: ShuffleTypes}
    lane = get_sub_group_local_id()
    src = lane + offset
    return sub_group_shuffle(val, ifelse(src <= get_max_sub_group_size(), src, lane))
end

@device_override function KI.shfl_up(val::T, offset::Integer) where {T <: ShuffleTypes}
    lane = get_sub_group_local_id()
    return sub_group_shuffle(val, ifelse(lane > offset, lane - offset, lane))
end

@device_override KI.shfl_xor(val::T, mask::Integer) where {T <: ShuffleTypes} =
    sub_group_shuffle_xor(val, mask)

@device_override KI.sub_group_any(pred::Bool) = SPIRVIntrinsics.sub_group_any(pred)

@device_override KI.sub_group_all(pred::Bool) = SPIRVIntrinsics.sub_group_all(pred)

@device_override function KI.sub_group_ballot(pred::Bool)
    mask = SPIRVIntrinsics.sub_group_ballot(pred)
    return UInt64(mask[1].value) | (UInt64(mask[2].value) << 32)
end

# The smaller of the width and the work-group size: a bound for loops over the lanes that isn't
# a constant, but is the same for all sub-groups of the work-group, which PoCL needs (see
# `POCLBackend`).
@inline function uniform_sub_group_bound()
    sz = KI.get_local_size(Int32)
    return min(KI.get_max_sub_group_size(Int32), sz.x * sz.y * sz.z)
end

# KernelInterface's fallback of `KI.sub_group_match_any` takes a step per distinct value, which
# differs between the sub-groups of a work-group. Take the same number of steps in all of them,
# as PoCL needs.
@device_override @inline function KI.sub_group_match_any(val)
    remaining = KI.sub_group_ballot(true)
    mask = zero(UInt64)
    step = Int32(0)
    bound = uniform_sub_group_bound()
    while step < bound
        done = remaining == zero(UInt64)
        leader = ifelse(done, Int32(1), trailing_zeros(remaining) % Int32 + Int32(1))
        other = KI.shfl(val, leader)
        same = !done & (val === other)
        group = KI.sub_group_ballot(same)
        mask = ifelse(same, group, mask)
        remaining &= ~group
        step += Int32(1)
    end
    return mask
end

# PoCL 7.2 miscompiles sub-group operations after a branch with an early exit (as bounds checks
# emit), because WorkitemLoops gives the peeled first work-item its own copy of their scratch
# memory: the native collectives below, and the unrolled shuffles of KernelInterface's
# fallbacks of `KI.sub_group_reduce` and `KI.sub_group_scan`. `pocl_standalone_jll` includes
# the fix (pocl/pocl#2239) since 7.2.1+1 (JuliaPackaging/Yggdrasil#15001).
const POCL_REPLICA_FIX = pkgversion(cl.pocl_standalone_jll) >= v"7.2.1+1"

@static if POCL_REPLICA_FIX
    # Native reductions and scans of `cl_khr_subgroups`, for `+` on 32- and 64-bit integers
    # and floats, and `min`/`max` on integers (OpenCL's `min` and `max` treat NaN and the sign
    # of zero differently from Julia's).
    const CollectiveIntTypes = Union{Int32, UInt32, Int64, UInt64}
    const CollectiveTypes = Union{CollectiveIntTypes, Float16, Float32, Float64}

    @device_override KI.sub_group_reduce(::typeof(+), val::CollectiveTypes) =
        SPIRVIntrinsics.sub_group_reduce_add(val)
    @device_override KI.sub_group_reduce(::typeof(min), val::CollectiveIntTypes) =
        SPIRVIntrinsics.sub_group_reduce_min(val)
    @device_override KI.sub_group_reduce(::typeof(max), val::CollectiveIntTypes) =
        SPIRVIntrinsics.sub_group_reduce_max(val)

    @device_override KI.sub_group_scan(::typeof(+), val::CollectiveTypes) =
        SPIRVIntrinsics.sub_group_scan_inclusive_add(val)
    @device_override KI.sub_group_scan(::typeof(min), val::CollectiveIntTypes) =
        SPIRVIntrinsics.sub_group_scan_inclusive_min(val)
    @device_override KI.sub_group_scan(::typeof(max), val::CollectiveIntTypes) =
        SPIRVIntrinsics.sub_group_scan_inclusive_max(val)
else
    # Without the fix, loop to `uniform_sub_group_bound()` rather than to the constant width.
    # Reduce by combining ranges of doubling length with `shfl_down`, skipping the lanes
    # without a work-item, and broadcast the result of the first lane.
    @device_override @inline function KI.sub_group_reduce(op, val)
        lane = KI.get_sub_group_local_id(Int32)
        sgsize = KI.get_sub_group_size(Int32)
        offset = Int32(1)
        bound = uniform_sub_group_bound()
        while offset < bound
            other = KI.shfl_down(val, offset)
            if lane + offset <= sgsize
                val = op(val, other)
            end
            offset <<= 1
        end
        return KI.shfl(val, 1)
    end

    # likewise for `KI.sub_group_scan`
    @device_override @inline function KI.sub_group_scan(op, val)
        lane = KI.get_sub_group_local_id(Int32)
        offset = Int32(1)
        bound = uniform_sub_group_bound()
        while offset < bound
            other = KI.shfl_up(val, offset)
            if lane > offset
                val = op(other, val)
            end
            offset <<= 1
        end
        return val
    end
end

@device_override @inline function KI._print(args...)
    POCL._print(args...)
end

end
