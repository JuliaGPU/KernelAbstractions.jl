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
# doesn't change during the kernel's execution, and that sub-groups are formed from
# consecutive work-items, x fastest, if the work-group is 1-D or its x extent is a multiple of
# the sub-group width (see the manual). Backends that can't ensure that don't report sub-group
# support.
#
# In a partial sub-group, the lanes `get_sub_group_size()+1:get_max_sub_group_size()` have no
# work-item: shuffles from them give unspecified values, and the votes, `sub_group_match_any`,
# `sub_group_reduce` and `sub_group_scan` only take the work-items of the sub-group into
# account.

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

"""
    shfl(val::T, lane::Integer)::T

Return `val` of the work-item with [`get_sub_group_local_id`](@ref) equal to `lane` in the
sub-group. When there is no such work-item, the result is an unspecified value (of type `T`).

All work-items of the sub-group have to execute `shfl` together (not in a divergent branch),
but they may read from different lanes.

`shfl` exchanges values, not memory: it is not a memory fence.

Types for which [`supports_shuffle`](@ref) returns `true` are supported. Besides the types a
backend supports natively, that includes other primitive types of 1, 2, 4, 8 or 16 bytes
(e.g. `Bool`, `Char` or `Int64`) if the backend supports `UInt32`, and `isbits` structs and
tuples of supported types, which are shuffled field by field. Primitive types of up to 4
bytes are shuffled as a `UInt32`, and larger ones as `UInt64` words, each of which is shuffled
natively if the backend supports `UInt64`, and as two `UInt32` words otherwise.

!!! note
    Backend implementations **must** implement this for the primitive types they support
    natively, which have to include `UInt32`, and only for those, so that other types reach
    the fallbacks:
    ```
    @device_override shfl(val::T, lane::Integer) where {T <: Union{...}}
    ```
"""
@inline shfl(val, lane::Integer) = shfl_fallback(x -> shfl(x, lane), val)

"""
    shfl_down(val::T, offset::Integer)::T

Return `val` of the work-item `offset` lanes further in the sub-group, i.e. with
[`get_sub_group_local_id`](@ref) equal to `get_sub_group_local_id() + offset`, for an `offset`
of at least 0. When that lane is past the sub-group width, i.e.
`get_sub_group_local_id() + offset > get_max_sub_group_size()`, the result is `val` of the
work-item itself, like CUDA's `shfl_down_sync`. When the lane is within the width but has no
work-item, in a partial sub-group, the result is an unspecified value (of type `T`).

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
@inline shfl_down(val, offset::Integer) = shfl_fallback(x -> shfl_down(x, offset), val)

"""
    shfl_up(val::T, offset::Integer)::T

Return `val` of the work-item `offset` lanes earlier in the sub-group, i.e. with
[`get_sub_group_local_id`](@ref) equal to `get_sub_group_local_id() - offset`, for an `offset`
of at least 0. When there is no such lane, i.e. `get_sub_group_local_id() <= offset`, the
result is `val` of the work-item itself, like CUDA's `shfl_up_sync`.

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
@inline shfl_up(val, offset::Integer) = shfl_fallback(x -> shfl_up(x, offset), val)

"""
    shfl_xor(val::T, mask::Integer)::T

Return `val` of the work-item whose 0-based lane id is the 0-based lane id of this work-item
xor `mask`, i.e. with [`get_sub_group_local_id`](@ref) equal to
`((get_sub_group_local_id() - 1) ⊻ mask) + 1`, for a `mask` between 0 and
`get_max_sub_group_size() - 1`. When that lane has no work-item, in a partial sub-group, the
result is an unspecified value (of type `T`).

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
@inline shfl_xor(val, mask::Integer) = shfl_fallback(x -> shfl_xor(x, mask), val)

# Shuffle a value of a type that the backend doesn't support natively, with `f` shuffling a
# value of a type it does support. The fallbacks are separate functions for primitive and for
# other types, so that the fallback of a struct with a field the backend doesn't support
# natively, e.g. an `Int64` on Metal, doesn't call itself, which inference gives up on (on
# Julia 1.10).
@inline function shfl_fallback(f, val::T) where {T}
    return isprimitivetype(T) ? shfl_words(f, val) : shfl_fields(f, val)
end

shfl_unsupported(T) = throw(
    ArgumentError(
        "Shuffling values of type $T is not supported by this backend, see `supports_shuffle`"
    )
)

# Whether a primitive type that a backend doesn't support natively can be shuffled as unsigned
# words
shuffle_as_words(T) = T !== UInt32 && sizeof(T) in (1, 2, 4, 8, 16)

# The unsigned integer type of the size of a primitive type `T`
const word_types = Dict(1 => UInt8, 2 => UInt16, 4 => UInt32, 8 => UInt64, 16 => UInt128)

# Shuffle a primitive value as unsigned words: values of up to 4 bytes are zero-extended to a
# `UInt32`. Other 8-byte values are shuffled as a `UInt64`, which the backend may support
# natively, and a `UInt64` it doesn't is split into two `UInt32` words. 16-byte values are
# split into two `UInt64` words.
@inline @generated function shfl_words(f, val::T) where {T}
    shuffle_as_words(T) || return :(shfl_unsupported($T))
    U = word_types[sizeof(T)]
    if sizeof(T) <= 4
        return :(reinterpret($T, f(reinterpret($U, val) % UInt32) % $U))
    elseif sizeof(T) == 8 && T !== UInt64
        return :(reinterpret($T, f(reinterpret(UInt64, val))))
    end
    W = sizeof(T) == 8 ? UInt32 : UInt64
    nbits = 8 * sizeof(W)
    words = (:((f((bits >> $(nbits * (i - 1))) % $W) % $U) << $(nbits * (i - 1))) for i in 1:2)
    return quote
        bits = reinterpret($U, val)
        return reinterpret($T, |($(words...)))
    end
end

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
    isbitstype(T) || return :(shfl_unsupported($T))
    return shfl_fields_expr(T, :val)
end

# The shuffles within segments of `width` lanes, implemented with `shfl` from a lane.

"""
    shfl(val::T, lane::Integer, width::Integer)::T
    shfl_down(val::T, offset::Integer, width::Integer)::T
    shfl_up(val::T, offset::Integer, width::Integer)::T
    shfl_xor(val::T, mask::Integer, width::Integer)::T

Shuffles within segments of `width` consecutive lanes of the sub-group, as if each segment
were a sub-group of its own: `lane` is the lane within the segment (between 1 and `width`, and
taken modulo `width` otherwise), and `shfl_down`, `shfl_up` and `shfl_xor` read from lanes of
the same segment. Where these would read from outside of the segment, they return `val` of the
work-item itself, like CUDA's shuffles with a `width` (and like `shfl_down` and `shfl_up`
without a `width` past the sub-group width). Reading from a lane of the segment that has no
work-item (in a partial sub-group) gives an unspecified value.

`width` has to be a power of two of at most the sub-group width
[`get_max_sub_group_size`](@ref), and the same for all work-items of the sub-group.

!!! note
    Backends **may** implement these, e.g. if they have native shuffles with a width. The
    fallbacks use [`shfl`](@ref) from a lane.
"""
@inline function shfl(val, lane::Integer, width::Integer)
    l0 = get_sub_group_local_id(Int32) - Int32(1)
    w = width % Int32
    base = l0 & ~(w - Int32(1))
    return shfl(val, base + ((lane % Int32 - Int32(1)) & (w - Int32(1))) + Int32(1))
end

@inline function shfl_down(val, offset::Integer, width::Integer)
    l0 = get_sub_group_local_id(Int32) - Int32(1)
    w = width % Int32
    d = offset % Int32
    src = ifelse((l0 & (w - Int32(1))) + d < w, l0 + d, l0)
    return shfl(val, src + Int32(1))
end

@inline function shfl_up(val, offset::Integer, width::Integer)
    l0 = get_sub_group_local_id(Int32) - Int32(1)
    w = width % Int32
    d = offset % Int32
    src = ifelse((l0 & (w - Int32(1))) >= d, l0 - d, l0)
    return shfl(val, src + Int32(1))
end

@inline function shfl_xor(val, mask::Integer, width::Integer)
    l0 = get_sub_group_local_id(Int32) - Int32(1)
    w = width % Int32
    x = l0 ⊻ (mask % Int32)
    src = ifelse((x & ~(w - Int32(1))) == (l0 & ~(w - Int32(1))), x, l0)
    return shfl(val, src + Int32(1))
end

"""
    sub_group_any(pred::Bool)::Bool

Whether `pred` is `true` for any work-item of the sub-group. All work-items of the sub-group
get the same result.

All work-items of the sub-group have to execute `sub_group_any` together (not in a divergent
branch).

It exchanges values, not memory: it is not a memory fence, see [`sub_group_barrier`](@ref).

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

It exchanges values, not memory: it is not a memory fence, see [`sub_group_barrier`](@ref).

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

It exchanges values, not memory: it is not a memory fence, see [`sub_group_barrier`](@ref).

!!! note
    Backend implementations that support sub-groups with a width of at most 64 **must**
    implement:
    ```
    @device_override sub_group_ballot(pred::Bool)::UInt64
    ```
"""
function sub_group_ballot end

"""
    sub_group_match_any(val)::UInt64

A mask of the work-items of the sub-group whose `val` is the same as the one of this
work-item (compared bitwise, with `===`), in the format of [`sub_group_ballot`](@ref). Work-items
with different values get different masks: e.g. `trailing_zeros(mask) + 1` is the lane of
the first work-item with the same value, and `count_ones(mask)` the number of them.

All work-items of the sub-group have to execute `sub_group_match_any` together (not in a
divergent branch). Values of the types that the shuffles support are supported, see
[`supports_shuffle`](@ref).

It exchanges values, not memory: it is not a memory fence, see [`sub_group_barrier`](@ref).

!!! note
    Backends **may** implement this, e.g. with CUDA's `match.any.sync`. The fallback finds
    the groups of equal values one by one with [`shfl`](@ref) and [`sub_group_ballot`](@ref),
    so it takes as many steps as there are distinct values.
"""
@inline function sub_group_match_any(val)
    remaining = sub_group_ballot(true)
    mask = zero(UInt64)
    # uniform: every step removes the group of the lowest remaining lane from `remaining`
    while remaining != zero(UInt64)
        leader = trailing_zeros(remaining) + 1
        same = val === shfl(val, leader)
        group = sub_group_ballot(same)
        mask = ifelse(same, group, mask)
        remaining &= ~group
    end
    return mask
end

# The votes within segments of `width` lanes, implemented with `sub_group_ballot`.

# The 0-based lane of the first work-item of this work-item's segment of `width` lanes
@inline segment_base(width) = (get_sub_group_local_id(Int32) - Int32(1)) & ~(width % Int32 - Int32(1))

# The bits of the segment of `width` lanes starting at `base` of a mask, moved to the bottom
@inline function segment_bits(mask::UInt64, base, width)
    w = width % UInt32
    bits = w >= UInt32(64) ? typemax(UInt64) : (UInt64(1) << w) - UInt64(1)
    return (mask >> (base % UInt32)) & bits
end

"""
    sub_group_any(pred::Bool, width::Integer)::Bool
    sub_group_all(pred::Bool, width::Integer)::Bool
    sub_group_ballot(pred::Bool, width::Integer)::UInt64
    sub_group_match_any(val, width::Integer)::UInt64

Votes within segments of `width` consecutive lanes of the sub-group, as if each segment were a
sub-group of its own (like the shuffles with a `width`, e.g. [`shfl`](@ref
shfl(::Any, ::Integer, ::Integer))): the work-items of a segment get the result for the
work-items of their segment. The masks of `sub_group_ballot` and `sub_group_match_any` have a
bit per lane of the segment: bit `i - 1` for its `i`-th lane, i.e. the lane
`get_sub_group_local_id()` with `(get_sub_group_local_id() - 1) % width == i - 1`. In a
partial sub-group, only the work-items of the segment take part.

`width` has to be a power of two of at most the sub-group width
[`get_max_sub_group_size`](@ref) (and of at most 64), and the same for all work-items of the
sub-group, which all have to execute the vote together (not in a divergent branch).

It exchanges values, not memory: it is not a memory fence, see [`sub_group_barrier`](@ref).

!!! note
    Backends **may** implement these. The fallbacks use [`sub_group_ballot`](@ref) and
    [`sub_group_match_any`](@ref) of the whole sub-group.
"""
@inline sub_group_ballot(pred::Bool, width::Integer) =
    segment_bits(sub_group_ballot(pred), segment_base(width), width)

@inline sub_group_any(pred::Bool, width::Integer) = sub_group_ballot(pred, width) != zero(UInt64)

@inline function sub_group_all(pred::Bool, width::Integer)
    # compare with the work-items of the segment, which may be partial
    base = segment_base(width)
    return segment_bits(sub_group_ballot(pred), base, width) ==
        segment_bits(sub_group_ballot(true), base, width)
end

@inline sub_group_match_any(val, width::Integer) =
    segment_bits(sub_group_match_any(val), segment_base(width), width)

"""
    sub_group_reduce(op, val::T)::T

Reduce `val` over the work-items of the sub-group with the associative binary operator `op`,
in the order of the lanes. All work-items of the sub-group get the result.

All work-items of the sub-group have to execute `sub_group_reduce` together (not in a
divergent branch). Values of the types that the shuffles support are supported, see
[`supports_shuffle`](@ref).

It exchanges values, not memory: it is not a memory fence, see [`sub_group_barrier`](@ref).

!!! note
    Backends **may** implement this for operators and types with a native reduction (e.g.
    `+` on `Float32`), dispatching on `typeof(op)`. The fallback combines ranges of doubling
    length with a butterfly of [`shfl_xor`](@ref), and in a partial sub-group broadcasts the
    result of the first lane with [`shfl`](@ref).
"""
@inline function sub_group_reduce(op, val)
    width = get_max_sub_group_size(Int32)
    sgsize = get_sub_group_size(Int32)
    lane0 = get_sub_group_local_id(Int32) - Int32(1)
    if ispow2(width)
        # A butterfly with `shfl_xor`, which unrolls for the constant width. Each step combines
        # the block of a work-item with the neighboring block, the lower one first, so that
        # only associativity is needed. In a full sub-group, every work-item ends up with the
        # reduction; in a partial one, blocks without work-items are skipped, which keeps the
        # result of the first lane correct, and it is broadcast. All sub-groups run the same
        # shuffles, since some backends (PoCL) need that across the sub-groups of a work-group.
        mask = Int32(1)
        while mask < width
            other = shfl_xor(val, mask)
            if (lane0 ⊻ mask) < sgsize
                if lane0 & mask == Int32(0)
                    val = op(val, other)
                else
                    val = op(other, val)
                end
            end
            mask <<= 1
        end
        first = shfl(val, 1)
        return ifelse(sgsize == width, val, first)
    else
        # combine ranges of doubling length with `shfl_down`, skipping the lanes without a
        # work-item, and broadcast the result of the first lane
        lane = lane0 + Int32(1)
        offset = Int32(1)
        while offset < width
            other = shfl_down(val, offset)
            if lane + offset <= sgsize
                val = op(val, other)
            end
            offset <<= 1
        end
        return shfl(val, 1)
    end
end

"""
    sub_group_scan(op, val::T)::T

The inclusive scan of `val` over the work-items of the sub-group with the associative binary
operator `op`, in the order of the lanes: the work-item in lane `i` gets the reduction of the
values of lanes `1` to `i`.

All work-items of the sub-group have to execute `sub_group_scan` together (not in a divergent
branch). Values of the types that the shuffles support are supported, see
[`supports_shuffle`](@ref).

It exchanges values, not memory: it is not a memory fence, see [`sub_group_barrier`](@ref).

!!! note
    Backends **may** implement this for operators and types with a native scan (e.g. `+` on
    `Float32`), dispatching on `typeof(op)`. The fallback is a Hillis-Steele scan with
    [`shfl_up`](@ref).
"""
@inline function sub_group_scan(op, val)
    lane = get_sub_group_local_id(Int32)
    # loop to the constant width, so that the loop unrolls: the lanes `shfl_up` reads from
    # always have a work-item, also in a partial sub-group
    width = get_max_sub_group_size(Int32)
    offset = Int32(1)
    while offset < width
        other = shfl_up(val, offset)
        if lane > offset
            val = op(other, val)
        end
        offset <<= 1
    end
    return val
end


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
