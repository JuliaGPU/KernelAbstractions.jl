## indexing

# The index queries are 1-based, and take the integer type `T` of their result. Backends
# implement the four primitive ones; the global ones have fallbacks derived from them.
#
# Supported `T` are the fixed-width integer types up to 64 bits. The operands are converted
# to `T` before any arithmetic, and the result is the exact value modulo `T` (as with `x % T`):
# a query never throws, and a value that doesn't fit wraps.

"""
    get_local_id([::Type{T}=Int])::@NamedTuple{x::T, y::T, z::T}

The 1-based index of the work-item within its work-group, as integers of type `T`.

`T` is a fixed-width integer type of at most 64 bits (e.g. `Int32` or `UInt64`); the result
is the exact value modulo `T`, as if computed with `x % T`.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_local_id(::Type{T})::@NamedTuple{x::T, y::T, z::T} where {T}
    ```
    The zero-argument form forwards to `get_local_id(Int)`.
"""
@inline get_local_id() = get_local_id(Int)

"""
    get_group_id([::Type{T}=Int])::@NamedTuple{x::T, y::T, z::T}

The 1-based index of the work-group within the launch, as integers of type `T`.

See [`get_local_id`](@ref) for the supported types `T`.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_group_id(::Type{T})::@NamedTuple{x::T, y::T, z::T} where {T}
    ```
    The zero-argument form forwards to `get_group_id(Int)`.
"""
@inline get_group_id() = get_group_id(Int)

"""
    get_local_size([::Type{T}=Int])::@NamedTuple{x::T, y::T, z::T}

The number of work-items in a work-group, as integers of type `T`.

See [`get_local_id`](@ref) for the supported types `T`.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_local_size(::Type{T})::@NamedTuple{x::T, y::T, z::T} where {T}
    ```
    The zero-argument form forwards to `get_local_size(Int)`.
"""
@inline get_local_size() = get_local_size(Int)

"""
    get_num_groups([::Type{T}=Int])::@NamedTuple{x::T, y::T, z::T}

The number of work-groups in the launch, as integers of type `T`.

See [`get_local_id`](@ref) for the supported types `T`.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_num_groups(::Type{T})::@NamedTuple{x::T, y::T, z::T} where {T}
    ```
    The zero-argument form forwards to `get_num_groups(Int)`.
"""
@inline get_num_groups() = get_num_groups(Int)

"""
    get_global_id([::Type{T}=Int])::@NamedTuple{x::T, y::T, z::T}

The 1-based index of the work-item within the launch, as integers of type `T`:
`(get_group_id(T) - 1) * get_local_size(T) + get_local_id(T)` per dimension.

See [`get_local_id`](@ref) for the supported types `T`.

!!! note
    The fallback derives this from the primitive queries. Backend implementations with a
    native builtin **should** override it, returning the same values:
    ```
    @device_override get_global_id(::Type{T})::@NamedTuple{x::T, y::T, z::T} where {T}
    ```
"""
@inline function get_global_id(::Type{T}) where {T}
    group = get_group_id(T)
    size = get_local_size(T)
    local_id = get_local_id(T)
    return (;
        x = (group.x - one(T)) * size.x + local_id.x,
        y = (group.y - one(T)) * size.y + local_id.y,
        z = (group.z - one(T)) * size.z + local_id.z,
    )
end
@inline get_global_id() = get_global_id(Int)

"""
    get_global_size([::Type{T}=Int])::@NamedTuple{x::T, y::T, z::T}

The number of work-items in the launch, as integers of type `T`:
`get_local_size(T) * get_num_groups(T)` per dimension. For an `ndrange` launch, this is the
`ndrange` padded to whole work-groups.

See [`get_local_id`](@ref) for the supported types `T`.

!!! note
    The fallback derives this from the primitive queries. Backend implementations with a
    native builtin **should** override it, returning the same values:
    ```
    @device_override get_global_size(::Type{T})::@NamedTuple{x::T, y::T, z::T} where {T}
    ```
"""
@inline function get_global_size(::Type{T}) where {T}
    size = get_local_size(T)
    groups = get_num_groups(T)
    return (; x = size.x * groups.x, y = size.y * groups.y, z = size.z * groups.z)
end
@inline get_global_size() = get_global_size(Int)


## sub-groups

# Sub-group support is optional, see `supports_subgroups`. A work-group is divided into
# sub-groups of `sub_group_size(backend)` work-items. How work-items are assigned to
# sub-groups is unspecified, except that `(get_sub_group_id(), get_sub_group_local_id())`
# is unique within a work-group and doesn't change during the kernel's execution.

"""
    get_sub_group_size([::Type{T}=Int])::T

The number of work-items in the sub-group: the sub-group width
([`get_max_sub_group_size`](@ref)), or fewer for the last sub-group of a work-group whose
size isn't a multiple of the width.

See [`get_local_id`](@ref) for the supported types `T`.

!!! note
    Backend implementations that support sub-groups **must** implement:
    ```
    @device_override get_sub_group_size(::Type{T})::T where {T}
    ```
    The zero-argument form forwards to `get_sub_group_size(Int)`.
"""
@inline get_sub_group_size() = get_sub_group_size(Int)

"""
    get_max_sub_group_size([::Type{T}=Int])::T

The sub-group width, [`sub_group_size(backend)`](@ref sub_group_size) on the host.

See [`get_local_id`](@ref) for the supported types `T`.

!!! note
    Backend implementations that support sub-groups **must** implement:
    ```
    @device_override get_max_sub_group_size(::Type{T})::T where {T}
    ```
    The zero-argument form forwards to `get_max_sub_group_size(Int)`.
"""
@inline get_max_sub_group_size() = get_max_sub_group_size(Int)

"""
    get_num_sub_groups([::Type{T}=Int])::T

The number of sub-groups in the work-group: `cld(prod(get_local_size()), get_max_sub_group_size())`.

See [`get_local_id`](@ref) for the supported types `T`.

!!! note
    Backend implementations that support sub-groups **must** implement:
    ```
    @device_override get_num_sub_groups(::Type{T})::T where {T}
    ```
    The zero-argument form forwards to `get_num_sub_groups(Int)`.
"""
@inline get_num_sub_groups() = get_num_sub_groups(Int)

"""
    get_sub_group_id([::Type{T}=Int])::T

The 1-based index of the sub-group within the work-group.

See [`get_local_id`](@ref) for the supported types `T`.

!!! note
    Backend implementations that support sub-groups **must** implement:
    ```
    @device_override get_sub_group_id(::Type{T})::T where {T}
    ```
    The zero-argument form forwards to `get_sub_group_id(Int)`.
"""
@inline get_sub_group_id() = get_sub_group_id(Int)

"""
    get_sub_group_local_id([::Type{T}=Int])::T

The 1-based index of the work-item within its sub-group (its lane). It doesn't depend on
which work-items of the sub-group are active, e.g. in a divergent branch.

See [`get_local_id`](@ref) for the supported types `T`.

!!! note
    Backend implementations that support sub-groups **must** implement:
    ```
    @device_override get_sub_group_local_id(::Type{T})::T where {T}
    ```
    The zero-argument form forwards to `get_sub_group_local_id(Int)`.
"""
@inline get_sub_group_local_id() = get_sub_group_local_id(Int)


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
    shfl_down(val::T, offset::Integer)::T

Return `val` of the work-item `offset` lanes further in the sub-group, i.e. with
[`get_sub_group_local_id`](@ref) equal to `get_sub_group_local_id() + offset`. When there is
no such work-item, the result is an unspecified value (of type `T`).

All work-items of the sub-group have to execute `shfl_down` together (not in a divergent
branch), with the same `offset`.

`shfl_down` exchanges values, not memory: it is not a memory fence.

!!! note
    Backend implementations **must** implement this for every `T` for which
    [`supports_shuffle`](@ref) returns `true`:
    ```
    @device_override shfl_down(val::T, offset::Integer) where T
    ```
"""
function shfl_down end


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

Like [`barrier`](@ref), for the work-items of a sub-group: wait until all work-items of the
sub-group have reached the barrier, and make their writes to global and local memory
before it visible to the sub-group.

All work-items of a sub-group have to reach the same `sub_group_barrier()`.

!!! note
    Backend implementations that support sub-groups **must** implement:
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
