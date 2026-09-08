"""
    get_global_size([::Type{T}=Type(Int)])::@NamedTuple{x::T, y::T, z::T}

Return the number of global work-items specified as a tuple of type `T`.
`T` defaults to `Int`.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_global_size(::Type{T})::@NamedTuple{x::T, y::T, z::T} where {T}
    ```
    The zero-argument form forwards to `get_global_size(Int)`.
"""
get_global_size() = get_global_size(Int)

"""
    get_global_id([::Type{T}=Type(Int)])::@NamedTuple{x::T, y::T, z::T}

Returns the unique global work-item ID as a tuple of type `T`. `T` defaults to `Int`.

!!! note
    1-based.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_global_id(::Type{T})::@NamedTuple{x::T, y::T, z::T} where {T}
    ```
    The zero-argument form forwards to `get_global_id(Int)`.
"""
get_global_id() = get_global_id(Int)

"""
    get_local_size([::Type{T}=Type(Int)])::@NamedTuple{x::T, y::T, z::T}

Return the number of local work-items specified as a tuple of type `T`.
`T` defaults to `Int`.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_local_size(::Type{T})::@NamedTuple{x::T, y::T, z::T} where {T}
    ```
    The zero-argument form forwards to `get_local_size(Int)`.
"""
get_local_size() = get_local_size(Int)

"""
    get_local_id([::Type{T}=Type(Int)])::@NamedTuple{x::T, y::T, z::T}

Returns the unique local work-item ID as a tuple of type `T`. `T` defaults to `Int`.

!!! note
    1-based.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_local_id(::Type{T})::@NamedTuple{x::T, y::T, z::T} where {T}
    ```
    The zero-argument form forwards to `get_local_id(Int)`.
"""
get_local_id() = get_local_id(Int)

"""
    get_num_groups([::Type{T}=Type(Int)])::@NamedTuple{x::T, y::T, z::T}

Returns the number of groups as a tuple of type `T`. `T` defaults to `Int`.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_num_groups(::Type{T})::@NamedTuple{x::T, y::T, z::T} where {T}
    ```
    The zero-argument form forwards to `get_num_groups(Int)`.
"""
get_num_groups() = get_num_groups(Int)

"""
    get_group_id([::Type{T}=Type(Int)])::@NamedTuple{x::T, y::T, z::T}

Returns the unique group ID as a tuple of type `T`. `T` defaults to `Int`.

!!! note
    1-based.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_group_id(::Type{T})::@NamedTuple{x::T, y::T, z::T} where {T}
    ```
    The zero-argument form forwards to `get_group_id(Int)`.
"""
get_group_id() = get_group_id(Int)

"""
    get_sub_group_size()::UInt32

Returns the number of work-items in the sub-group.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_sub_group_size()::UInt32
    ```
"""
function get_sub_group_size end

"""
    get_max_sub_group_size()::UInt32

Returns the maximum sub-group size for sub-groups in the current workgroup.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_max_sub_group_size()::UInt32
    ```
"""
function get_max_sub_group_size end

"""
    get_num_sub_groups()::UInt32

Returns the number of sub-groups in the current workgroup.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_num_sub_groups()::UInt32
    ```
"""
function get_num_sub_groups end

"""
    get_sub_group_id()::UInt32

Returns the sub-group ID within the work-group.

!!! note
    1-based.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_sub_group_id()::UInt32
    ```
"""
function get_sub_group_id end

"""
    get_sub_group_local_id()::UInt32

Returns the work-item ID within the current sub-group.

!!! note
    1-based.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_sub_group_local_id()::UInt32
    ```
"""
function get_sub_group_local_id end


"""
    localmemory(::Type{T}, dims)

Declare memory that is local to a workgroup.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override localmemory(::Type{T}, ::Val{Dims}) where {T, Dims}
    ```
    As well as the on-device functionality.
"""
localmemory(::Type{T}, dims) where {T} = localmemory(T, Val(dims))

# The `Val` form only exists in a backend's overlay method table, so off-device it
# would otherwise fall back to the forwarding method above and recurse forever.
localmemory(::Type{T}, ::Val) where {T} =
    error("Local memory used outside kernel or not captured")


"""
    shfl_down(val::T, offset::Integer) where T

Read `val` from a lane with higher id given by `offset`.

!!! note
    `shfl_down` must be encountered by all workitems of a sub-group executing the kernel or by none at all.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override shfl_down(val::T, offset::Integer) where T
    ```
    As well as the on-device functionality.

    This implementation **must** be synchronizing.
    That is, kernels using this function can safely assume that
    they do **not** need a `sub_group_barrier` before calling
    this function.
"""
function shfl_down end

"""
    shfl_down_types(::Backend)::Vector{DataType}

Returns a vector of `DataType`s supported on `backend`

!!! note
    Backend implementations **must** implement this function
    only if they support `shfl_down` for any types.
"""
shfl_down_types(::Backend) = DataType[]


"""
    barrier()

After a `barrier()` call, all read and writes to global and local memory
from each thread in the workgroup are visible in from all other threads in the
workgroup.

This does **not** guarantee that a write from a thread in a certain workgroup will
be visible to a thread in a different workgroup.

!!! note
    `barrier()` must be encountered by all workitems of a work-group executing the kernel or by none at all.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override barrier()
    ```
"""
function barrier()
    error("Group barrier used outside kernel or not captured")
end

"""
    sub_group_barrier()

After a `sub_group_barrier()` call, all read and writes to global and local memory
from each thread in the sub-group are visible in from all other threads in the
sub-group.

This does **not** guarantee that a write from a thread in a certain sub-group will
be visible to a thread in a different sub-group.

!!! note
    `sub_group_barrier()` must be encountered by all workitems of a sub-group executing the kernel or by none at all.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override sub_group_barrier()
    ```
"""
function sub_group_barrier()
    error("Sub-group barrier used outside kernel or not captured")
end

"""
    _print(args...)

    Overloaded by backends to enable `KernelAbstractions.@print`
    functionality.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override _print(args...)
    ```
    If the backend does not support printing,
    define it to return `nothing`.

The generic fallback prints on the host, which keeps CPU backends working.
`Val` arguments are unwrapped, since `KernelAbstractions.@print` uses them to
pass literal strings through to backends that require compile-time format strings.
"""
@generated function _print(items...)
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
