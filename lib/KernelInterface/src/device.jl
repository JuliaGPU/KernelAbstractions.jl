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
# sub-groups of at most `sub_group_size(backend)` work-items. Which work-items form a
# sub-group, how many sub-groups there are and which are partial is unspecified, except
# that `(get_sub_group_id(), get_sub_group_local_id())` is unique within a work-group and
# doesn't change during the kernel's execution, and that a 1-D work-group of at most
# `sub_group_size(backend)` work-items is a single sub-group. Backends that can't ensure
# that don't report sub-group support. See the manual.

"""
    get_sub_group_size([::Type{T}=Int])::T

The number of work-items in the sub-group, at most the sub-group width
([`get_max_sub_group_size`](@ref)). Which sub-groups have fewer work-items than the width
is unspecified: when the work-group size isn't a multiple of the width, there can be more
than one, e.g. one per row of a multi-dimensional work-group.

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

The sub-group width (the warp or wavefront size), [`sub_group_size(backend)`](@ref
sub_group_size) on the host.

It is a compile-time constant of the generated code, so code that depends on it, e.g. a
loop over the lanes or a shuffle butterfly, is specialized for it. It isn't known during
type inference, though: to pick types or `Val` parameters from the width, e.g. the integer
type of a mask with a bit per lane, pass [`sub_group_size(backend)`](@ref sub_group_size)
from the host.

See [`get_local_id`](@ref) for the supported types `T`.

!!! note
    Backend implementations that support sub-groups **must** implement this, returning the
    width the kernel is compiled for as a constant (not by querying the device at run
    time):
    ```
    @device_override get_max_sub_group_size(::Type{T})::T where {T}
    ```
    The zero-argument form forwards to `get_max_sub_group_size(Int)`.
"""
@inline get_max_sub_group_size() = get_max_sub_group_size(Int)

"""
    get_num_sub_groups([::Type{T}=Int])::T

The number of sub-groups in the work-group. It is at least
`cld(prod(get_local_size()), get_max_sub_group_size())`, but can be larger, since more than
one sub-group can be partial. Size storage for a value per sub-group for up to one
sub-group per work-item.

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

The 1-based index of the sub-group within the work-group, between 1 and
[`get_num_sub_groups`](@ref). How it relates to [`get_local_id`](@ref) is unspecified.

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

The 1-based index of the work-item within its sub-group (its lane), between 1 and
[`get_sub_group_size`](@ref). It doesn't depend on which work-items of the sub-group are
active, e.g. in a divergent branch.

See [`get_local_id`](@ref) for the supported types `T`.

!!! note
    Backend implementations that support sub-groups **must** implement:
    ```
    @device_override get_sub_group_local_id(::Type{T})::T where {T}
    ```
    The zero-argument form forwards to `get_sub_group_local_id(Int)`.
"""
@inline get_sub_group_local_id() = get_sub_group_local_id(Int)


## memory

"""
    localmemory(::Type{T}, dims)

Declare an array of element type `T` and size `dims` in memory that is local to a
work-group. `dims` has to be known at compile time.

Every call site of `localmemory` in a kernel has its own memory, shared by all work-items
of a work-group. It is uninitialized, and lives until the work-group finishes. Executing the
same call site again, e.g. in a loop, returns the same memory. A function containing a call
that is itself called from several places may get the same memory at each of them, or
different memory, depending on whether it is inlined: don't rely on either. Use
[`barrier`](@ref) to make writes visible to the other work-items.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override localmemory(::Type{T}, ::Val{Dims}) where {T, Dims}
    ```
"""
localmemory(::Type{T}, dims) where {T} = localmemory(T, Val(dims))

# The `Val` form only exists in a backend's overlay method table, so off-device it
# would otherwise fall back to the forwarding method above and recurse forever.
localmemory(::Type{T}, ::Val) where {T} =
    error("Local memory used outside kernel or not captured")


## communication

# Shuffles exchange values between the work-items of a sub-group. Backends implement them for
# the primitive types for which `supports_shuffle` returns `true`; the fallbacks below
# shuffle other `isbits` types field by field.

"""
    shfl(val::T, lane::Integer)::T

Return `val` of the work-item with [`get_sub_group_local_id`](@ref) equal to `lane` in the
sub-group. When there is no such work-item, the result is an unspecified value (of type `T`).

All work-items of the sub-group have to execute `shfl` together (not in a divergent branch),
but they may read from different lanes.

`shfl` exchanges values, not memory: it is not a memory fence.

Types for which [`supports_shuffle`](@ref) returns `true` are supported, as well as `isbits`
structs and tuples of such types, which are shuffled field by field.

!!! note
    Backend implementations **must** implement this for every primitive type `T` for which
    [`supports_shuffle`](@ref) returns `true`, and only for those, so that other types reach
    the fallback that shuffles structs field by field:
    ```
    @device_override shfl(val::T, lane::Integer) where {T <: Union{...}}
    ```
"""
@inline shfl(val, lane::Integer) = shfl_fields(x -> shfl(x, lane), val)

"""
    shfl_down(val::T, offset::Integer)::T

Return `val` of the work-item `offset` lanes further in the sub-group, i.e. with
[`get_sub_group_local_id`](@ref) equal to `get_sub_group_local_id() + offset`. When there is
no such work-item, the result is an unspecified value (of type `T`).

All work-items of the sub-group have to execute `shfl_down` together (not in a divergent
branch), with the same `offset`.

`shfl_down` exchanges values, not memory: it is not a memory fence. See [`shfl`](@ref) for
the supported types.

!!! note
    Backend implementations **must** implement this like [`shfl`](@ref):
    ```
    @device_override shfl_down(val::T, offset::Integer) where {T <: Union{...}}
    ```
"""
@inline shfl_down(val, offset::Integer) = shfl_fields(x -> shfl_down(x, offset), val)

"""
    shfl_up(val::T, offset::Integer)::T

Return `val` of the work-item `offset` lanes earlier in the sub-group, i.e. with
[`get_sub_group_local_id`](@ref) equal to `get_sub_group_local_id() - offset`. When there is
no such work-item, the result is an unspecified value (of type `T`).

All work-items of the sub-group have to execute `shfl_up` together (not in a divergent
branch), with the same `offset`.

`shfl_up` exchanges values, not memory: it is not a memory fence. See [`shfl`](@ref) for
the supported types.

!!! note
    Backend implementations **must** implement this like [`shfl`](@ref):
    ```
    @device_override shfl_up(val::T, offset::Integer) where {T <: Union{...}}
    ```
"""
@inline shfl_up(val, offset::Integer) = shfl_fields(x -> shfl_up(x, offset), val)

"""
    shfl_xor(val::T, mask::Integer)::T

Return `val` of the work-item whose 0-based lane id is the 0-based lane id of this work-item
xor `mask`, i.e. with [`get_sub_group_local_id`](@ref) equal to
`((get_sub_group_local_id() - 1) ⊻ mask) + 1`. When there is no such work-item, the result is
an unspecified value (of type `T`).

All work-items of the sub-group have to execute `shfl_xor` together (not in a divergent
branch), with the same `mask`. A butterfly over the masks `width ÷ 2, …, 2, 1` (for the
sub-group width [`get_max_sub_group_size`](@ref)) reduces a full sub-group such that every
work-item gets the result.

`shfl_xor` exchanges values, not memory: it is not a memory fence. See [`shfl`](@ref) for
the supported types.

!!! note
    Backend implementations **must** implement this like [`shfl`](@ref):
    ```
    @device_override shfl_xor(val::T, mask::Integer) where {T <: Union{...}}
    ```
"""
@inline shfl_xor(val, mask::Integer) = shfl_fields(x -> shfl_xor(x, mask), val)

# The expression that shuffles `ex::S` field by field, calling `f` on the primitive fields
function shfl_fields_expr(S, ex)
    isprimitivetype(S) && return :(f($ex))
    fields = (shfl_fields_expr(fieldtype(S, i), :(getfield($ex, $i))) for i in 1:fieldcount(S))
    return Expr(:new, S, fields...)
end

# Shuffle a value that the backend doesn't support directly field by field. Nested fields are
# unrolled here, rather than shuffled with a recursive call, which inference gives up on
# (on Julia 1.10), so that `f` is only called on the primitive types.
@inline @generated function shfl_fields(f, val::T) where {T}
    if !isbitstype(T) || isprimitivetype(T)
        return :(
            throw(
                ArgumentError(
                    $("Shuffling values of type $T is not supported by this backend, see `supports_shuffle`")
                )
            )
        )
    end
    return shfl_fields_expr(T, :val)
end

"""
    sub_group_any(pred::Bool)::Bool

Whether `pred` is `true` for any work-item of the sub-group. All work-items of the sub-group
get the same result.

All work-items of the sub-group have to execute `sub_group_any` together (not in a divergent
branch).

!!! note
    Backend implementations that support sub-groups **must** implement:
    ```
    @device_override sub_group_any(pred::Bool)::Bool
    ```
"""
function sub_group_any end

"""
    sub_group_all(pred::Bool)::Bool

Whether `pred` is `true` for all work-items of the sub-group. All work-items of the
sub-group get the same result.

All work-items of the sub-group have to execute `sub_group_all` together (not in a divergent
branch).

!!! note
    Backend implementations that support sub-groups **must** implement:
    ```
    @device_override sub_group_all(pred::Bool)::Bool
    ```
"""
function sub_group_all end

"""
    sub_group_ballot(pred::Bool)::UInt64

A mask of the work-items of the sub-group for which `pred` is `true`: bit `i - 1` (counting
from the least significant bit) is set for the work-item with
[`get_sub_group_local_id`](@ref) equal to `i`. All work-items of the sub-group get the same
result. Use e.g. `count_ones` to count the work-items, or `trailing_zeros` to find the first
one.

All work-items of the sub-group have to execute `sub_group_ballot` together (not in a
divergent branch). Only sub-groups of at most 64 work-items are supported.

!!! note
    Backend implementations that support sub-groups with a width of at most 64 **must**
    implement:
    ```
    @device_override sub_group_ballot(pred::Bool)::UInt64
    ```
"""
function sub_group_ballot end


## synchronization

"""
    barrier()

Wait until all work-items of the work-group have reached the barrier. Afterwards, the
writes to global and local memory that each work-item made before the barrier are visible
to all work-items of the work-group.

This does **not** order memory between work-groups.

All work-items of a work-group have to reach the same `barrier()` (not in a divergent branch).

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


## printing

"""
    _print(args...)

Print `args` from a kernel; the backend hook behind `KernelAbstractions.@print`.

!!! note
    Backend implementations **should** implement:
    ```
    @device_override _print(args...)
    ```
    A backend that can't print from a kernel defines it to return `nothing`, and
    documents that.

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
