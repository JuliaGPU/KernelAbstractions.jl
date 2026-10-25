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

    if !cl.pocl_standalone_jll.is_available()
        println(io, "POCL is not available for this platform.")
        return
    end
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
# its work-items are iterations of a loop, which is faster than dividing for any group size
KA.fold_threshold(::POCLBackend) = 1

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
    # the sub-group width defaults to the one `KI.sub_group_size` reports. it is determined
    # when creating the compiler configuration, since passing it here would make the keyword
    # arguments' type depend on the device, and the call to `clfunction` dynamic.
    kernel = clfunction(clconvert(f), tt; name, kwargs...)
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

KI.max_work_group_size(kernel::KI.Kernel{<:POCLBackend})::Int =
    kernel.kern.kernel.fun.work_group_size
KI.max_work_group_size(::POCLBackend)::Int = device_limits().max_work_group_size
KI.max_work_group_dims(::POCLBackend)::NTuple{3, Int} = device_limits().max_work_group_dims
# the grid is only limited by the size of `size_t`
KI.max_num_groups(::POCLBackend)::NTuple{3, Int} = (typemax(Int), typemax(Int), typemax(Int))
KI.sub_group_size(::POCLBackend)::Int = device_limits().sub_group_size
KI.multiprocessor_count(::POCLBackend)::Int = device_limits().compute_units

# PoCL runs each work-group on one of its threads, looping over (and vectorizing across) the
# work-items. the largest work-group would run a launch of up to that many work-items on a
# single thread, so split it into about as many work-groups as there are threads, as long as
# the work-groups stay large enough to vectorize. this assumes that the work is balanced
# across the work-items.
KI.launch_configuration(
    kernel::KI.Kernel{<:POCLBackend}; nitems::Union{Integer, Nothing} = nothing,
    max_work_group_size::Integer = typemax(Int)
) = (; workgroupsize = launch_workgroupsize(kernel.kern.kernel.fun, nitems, max_work_group_size))

# kept out of `KI.launch_configuration`, which is compiled for every kernel
@noinline function launch_workgroupsize(kernel::cl.Kernel, nitems, max_work_group_size)
    max_items = min(kernel.work_group_size, max_work_group_size)
    items = cpu_workgroupsize(
        nitems, device_limits().compute_units, max_items, kernel.preferred_work_group_size_multiple
    )
    return Int(items)
end

# PoCL's own choice when a launch doesn't give a local size (`pocl_default_local_size_optimizer`
# in `lib/CL/devices/pocl_local_size.c`), in one dimension: start from the largest work-group,
# and split it while there are at most half as many work-groups as threads, but not below
# 4 times the preferred work-group size multiple. two differences: PoCL only uses local sizes
# that divide the global size; KernelAbstractions supports partial work-groups, so this uses
# powers of two instead, which divide the extents of `ndrange`s that are powers of two. and
# PoCL doesn't split at all if splitting over all threads would go below the minimum, which
# leaves e.g. 1000 work-items on 64 threads in one work-group; this splits down to the
# minimum instead.
function cpu_workgroupsize(nitems, threads, max_items, multiple)
    nitems === nothing && return max_items
    nitems = max(nitems, 1)
    min_items = min(4 * max(multiple, 1), max_items)
    items = min(nextpow(2, min(nitems, max_items)), max_items)
    groups = cld(nitems, items)
    while (splits = min(threads ÷ groups, items ÷ min_items)) > 1
        # PoCL splits by the largest factor of at most `splits` that divides the work-group
        items ÷= prevpow(2, splits)
        groups = cld(nitems, items)
    end
    return items
end

KI.supports_subgroups(::POCLBackend) = device_limits().sub_group_size > 0
# PoCL forms sub-groups from consecutive work-items, but executes sub-group operations as
# work-group barriers, so they need work-group-uniform control flow (see `POCLBackend`)
KI.supports_linear_subgroups(backend::POCLBackend) = KI.supports_subgroups(backend)
KI.supports_independent_subgroups(::POCLBackend) = false
# PoCL limitation: for the same reason, sub-groups can't be partitioned into segments that
# diverge
KI.supports_partitionable_subgroups(::POCLBackend) = false

# The types `sub_group_shuffle` supports; other types are shuffled as words. `Float16` and
# `Float64` are reported as unsupported on devices without `cl_khr_fp16` or `cl_khr_fp64`,
# rather than shuffled as words: the overrides below are selected by type, not by device.
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
    SPIRVIntrinsics.sub_group_shuffle(val, lane)

# past the sub-group width, `shfl_down` and `shfl_up` return the work-item's own value, which
# `sub_group_shuffle` (like SPIR-V's `OpGroupNonUniformShuffleDown`) leaves undefined
@device_override function KI.shfl_down(val::T, offset::Integer) where {T <: ShuffleTypes}
    lane = get_sub_group_local_id()
    # compared before adding, so that large offsets don't overflow
    inside = offset <= get_max_sub_group_size() - lane
    return SPIRVIntrinsics.sub_group_shuffle(val, ifelse(inside, lane + offset, lane))
end

@device_override function KI.shfl_up(val::T, offset::Integer) where {T <: ShuffleTypes}
    lane = get_sub_group_local_id()
    return SPIRVIntrinsics.sub_group_shuffle(val, ifelse(lane > offset, lane - offset, lane))
end

@device_override KI.shfl_xor(val::T, mask::Integer) where {T <: ShuffleTypes} =
    SPIRVIntrinsics.sub_group_shuffle_xor(val, mask)

# Every shuffle is a work-group barrier, so the words of a packed struct are exchanged with
# vector shuffles (`POCL.vector_shuffle`), a barrier per up to 16 words rather than per word.
# The last vector is padded to a size that exists (2, 3, 4, 8 or 16 words).
@generated function shuffle_word_vectors(words::NTuple{N, UInt32}, lane::Int32) where {N}
    body = Any[Expr(:meta, :inline)]
    results = Any[]
    for first in 1:16:N
        k = min(N - first + 1, 16)
        if k == 1
            push!(results, :(KI.shfl(words[$first], lane)))
            continue
        end
        n = k <= 4 ? k : k <= 8 ? 8 : 16
        elements = ((j < k ? :(VecElement(words[$(first + j)])) : :(VecElement(UInt32(0)))) for j in 0:(n - 1))
        v = Symbol(:v, first)
        push!(body, :($v = POCL.vector_shuffle(($(elements...),), lane)))
        append!(results, (:($v[$j].value) for j in 1:k))
    end
    push!(body, :(($(results...),)))
    return Expr(:block, body...)
end

@device_override @inline KI.shfl_word_tuple(op::KI.ShuffleOp, words::NTuple{N, UInt32}) where {N} =
    shuffle_word_vectors(words, KI.shuffle_source(op))

@device_override KI.sub_group_any(pred::Bool) = SPIRVIntrinsics.sub_group_any(pred)

@device_override KI.sub_group_all(pred::Bool) = SPIRVIntrinsics.sub_group_all(pred)

@device_override function KI.sub_group_ballot(pred::Bool)
    mask = SPIRVIntrinsics.sub_group_ballot(pred)
    return UInt64(mask[1].value) | (UInt64(mask[2].value) << 32)
end

# Native reductions and scans of `cl_khr_subgroups`, for `+` on 32- and 64-bit integers and
# floats, and `min`/`max` on integers (OpenCL's `min` and `max` treat NaN and the sign of zero
# differently from Julia's). They need the fix of PoCL 7.2's peeling of the first work-item
# (pocl/pocl#2239), which `pocl_standalone_jll` includes since 7.2.1+1.
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

# the exclusive scans of `cl_khr_subgroups` start from the identity, not from `init`
@device_override function KI.sub_group_exclusive_scan(::typeof(+), val::T, init::T) where {T <: CollectiveTypes}
    prefix = SPIRVIntrinsics.sub_group_scan_exclusive_add(val)
    return ifelse(get_sub_group_local_id() == 1, init, init + prefix)
end

@device_override @inline function KI._print(args...)
    POCL._print(args...)
end

end
