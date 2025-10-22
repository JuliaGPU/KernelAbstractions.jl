module KernelIntrinsics

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

function barrier()
    error("Group barrier used outside kernel or not captured")
end
function print end


"""
    KIKernel{Backend, BKern}

KIKernel closure struct that is used to represent the backend
kernel on the host.

!!! note
    Backend implementations **must** implement:
    ```
    KI.KIKernel(::NewBackend, f, args...; kwargs...)
    (kernel::KIKernel{<:NewBackend})(args...; numworkgroups=nothing, workgroupsize=nothing)
    ```
    As well as the on-device functionality.
"""
struct KIKernel{Backend, BKern}
    backend::Backend
    kern::BKern
end

function kernel_max_work_group_size end
function max_work_group_size end
function multiprocessor_count end
end
