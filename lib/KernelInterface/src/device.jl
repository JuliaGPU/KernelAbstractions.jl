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
# sub-group, how many sub-groups there are and which are partial is unspecified (unless
# `supports_linear_subgroups`), except that `(get_sub_group_id(), get_sub_group_local_id())`
# is unique within a work-group and doesn't change during the kernel's execution (see the
# manual).
#
# In a partial sub-group, the lanes `get_sub_group_size()+1:get_max_sub_group_size()` have no
# work-item: shuffles from them give unspecified values, and the votes, `sub_group_reduce` and
# the scans only take the work-items of the sub-group into account.

"""
    get_sub_group_size([::Type{T}=Int])::T

The number of work-items in the sub-group, at most the sub-group width
([`get_max_sub_group_size`](@ref)). Which sub-groups have fewer work-items than the width
is unspecified (unless [`supports_linear_subgroups`](@ref)): there can be more than one,
e.g. one per row of a multi-dimensional work-group.

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

Backends should make it a constant of the generated code, so that code depending on it, e.g.
a loop over the lanes or a shuffle butterfly, is specialized for it; on some, it is only
folded late in compilation, or not at all. It is never known during type inference: to pick
types or `Val` parameters from the width, e.g. the integer type of a mask with a bit per
lane, pass [`sub_group_size(backend)`](@ref sub_group_size) from the host.

See [`get_local_id`](@ref) for the supported types `T`.

!!! note
    Backend implementations that support sub-groups **must** implement this, and **should**
    return the width the kernel is compiled for as a constant rather than query the device
    at run time, so that code depending on it is specialized for the width:
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
one sub-group can be partial (unless [`supports_linear_subgroups`](@ref)). Size storage for a
value per sub-group for up to one sub-group per work-item.

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
[`get_num_sub_groups`](@ref). How it relates to [`get_local_id`](@ref) is unspecified,
unless [`supports_linear_subgroups`](@ref).

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
active, e.g. in a divergent branch. How it relates to [`get_local_id`](@ref) is unspecified,
unless [`supports_linear_subgroups`](@ref).

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

# The shuffles, votes, collectives and `sub_group_barrier` have to be executed by all
# work-items of a sub-group together, in control flow that is uniform over the work-group
# unless `supports_independent_subgroups`. The id and size queries above can be used anywhere.

"""
    shfl(val::T, lane::Integer)::T

Return `val` of the work-item with [`get_sub_group_local_id`](@ref) equal to `lane` in the
sub-group. When there is no such work-item, the result is an unspecified value (of type `T`).

All work-items of the sub-group have to execute `shfl` together (not in a divergent branch),
but they may read from different lanes. Unless [`supports_independent_subgroups`](@ref), all
sub-groups of the work-group have to execute it.

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
work-item itself, as on CUDA, HIP and Metal. When the lane is within the width but has no
work-item, in a partial sub-group, the result is an unspecified value (of type `T`).

All work-items of the sub-group have to execute `shfl_down` together (not in a divergent
branch), with the same `offset`; see [`shfl`](@ref).

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
result is `val` of the work-item itself, as on CUDA, HIP and Metal.

All work-items of the sub-group have to execute `shfl_up` together (not in a divergent
branch), with the same `offset`; see [`shfl`](@ref).

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
`((get_sub_group_local_id() - 1) ⊻ mask) + 1`. `mask` has to be between 0 and
`get_max_sub_group_size() - 1`. When that lane has no work-item, in a partial sub-group, the
result is an unspecified value (of type `T`).

All work-items of the sub-group have to execute `shfl_xor` together (not in a divergent
branch), with the same `mask`; see [`shfl`](@ref). A butterfly over the masks `width ÷ 2, …, 2, 1` (for the
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

# The unsigned integer type of the size of a primitive type `T` (of 1, 2, 4, 8 or 16 bytes)
const word_types = (UInt8, UInt16, UInt32, UInt64, UInt128)
word_type(T) = word_types[trailing_zeros(sizeof(T)) + 1]

# Shuffle a primitive value as unsigned words: values of up to 4 bytes are zero-extended to a
# `UInt32`. Other 8-byte values are shuffled as a `UInt64`, which the backend may support
# natively, and a `UInt64` it doesn't is split into two `UInt32` words. 16-byte values are
# split into two `UInt64` words.
@inline @generated function shfl_words(f, val::T) where {T}
    shuffle_as_words(T) || return :(shfl_unsupported($T))
    U = word_type(T)
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

# A `UInt64` that the backend doesn't shuffle natively is split into two `UInt32` words by a
# method of its own, so that another 8-byte type, shuffled as a `UInt64`, doesn't reach the
# fallback through itself, which inference gives up on (on Julia 1.10). Backends that shuffle
# `UInt64` natively override these.
@inline shfl(val::UInt64, lane::Integer) = shfl_words(x -> shfl(x, lane), val)
@inline shfl_down(val::UInt64, offset::Integer) = shfl_words(x -> shfl_down(x, offset), val)
@inline shfl_up(val::UInt64, offset::Integer) = shfl_words(x -> shfl_up(x, offset), val)
@inline shfl_xor(val::UInt64, mask::Integer) = shfl_words(x -> shfl_xor(x, mask), val)

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

Shuffles within segments of `width` consecutive lanes of the sub-group, with CUDA's semantics
for a `width`:

- `shfl` reads from lane `lane` of the caller's segment, with `lane` taken modulo `width`
  (lane `width + 1` is the segment's first lane);
- `shfl_down` and `shfl_up` read from the lane `offset` lanes further or earlier, and return
  `val` of the work-item itself where that lane is outside of the caller's segment;
- `shfl_xor` reads from the lane whose 0-based position in the segment is the caller's xor
  `mask`, which has to be between 0 and `width - 1`.

Reading from a lane of the segment that has no work-item (in a partial sub-group) gives an
unspecified value. `width` has to be a power of two of at most the sub-group width
[`get_max_sub_group_size`](@ref), and the same for all work-items of the sub-group.

The `width` only changes which lanes are read from: as for the shuffles without a `width`, all
work-items of the sub-group have to execute them together, not just those of a segment.

!!! note
    Backends **may** implement these, e.g. if they have native shuffles with a width. The
    fallbacks use [`shfl`](@ref) from a lane.
"""
@inline function shfl(val, lane::Integer, width::Integer)
    l0 = get_sub_group_local_id(Int32) - Int32(1)
    w = width % Int32
    base = l0 & ~(w - Int32(1))
    # `lane - 1` modulo the power of two `width`, before narrowing `lane`
    return shfl(val, base + ((lane - 1) & (width - 1)) % Int32 + Int32(1))
end

# The offsets are compared with the width before they are narrowed to `Int32`, so that large
# offsets read from outside of the segment.
@inline function shfl_down(val, offset::Integer, width::Integer)
    l0 = get_sub_group_local_id(Int32) - Int32(1)
    w = width % Int32
    pos = l0 & (w - Int32(1))
    inside = offset < width - pos
    src = ifelse(inside, l0 + offset % Int32, l0)
    return shfl(val, src + Int32(1))
end

@inline function shfl_up(val, offset::Integer, width::Integer)
    l0 = get_sub_group_local_id(Int32) - Int32(1)
    w = width % Int32
    pos = l0 & (w - Int32(1))
    inside = offset <= pos
    src = ifelse(inside, l0 - offset % Int32, l0)
    return shfl(val, src + Int32(1))
end

@inline function shfl_xor(val, mask::Integer, width::Integer)
    l0 = get_sub_group_local_id(Int32) - Int32(1)
    return shfl(val, (l0 ⊻ (mask % Int32)) + Int32(1))
end

"""
    sub_group_any(pred::Bool)::Bool

Whether `pred` is `true` for any work-item of the sub-group. All work-items of the sub-group
get the same result.

All work-items of the sub-group have to execute `sub_group_any` together (not in a divergent
branch); see [`shfl`](@ref).

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
branch); see [`shfl`](@ref).

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
from the least significant bit) is set if and only if the sub-group has a work-item with
[`get_sub_group_local_id`](@ref) equal to `i` and its `pred` is `true`. All other bits, e.g.
those of lanes without a work-item in a partial sub-group, are zero. All work-items of the
sub-group get the same result. Use e.g. `count_ones` to count the work-items, or
`trailing_zeros` to find the first one.

All work-items of the sub-group have to execute `sub_group_ballot` together (not in a
divergent branch); see [`shfl`](@ref). Only sub-groups of at most 64 work-items are supported.

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
    sub_group_reduce(op, val::T)::T

Reduce `val` over the work-items of the sub-group with the binary operator `op`, in the order
of the lanes: `op(…op(op(v₁, v₂), v₃)…, vₙ)` for the values `vᵢ` of lanes `1:n`, with
`n = get_sub_group_size()`, up to associativity. `op` has to be associative, but needn't be
commutative. All work-items of the sub-group get the result.

All work-items of the sub-group have to execute `sub_group_reduce` together (not in a
divergent branch), with the same `op`; see [`shfl`](@ref). Values of the types that the
shuffles support are supported, see [`supports_shuffle`](@ref), and others for which the
backend has a native reduction.

It exchanges values, not memory: it is not a memory fence, see [`sub_group_barrier`](@ref).

!!! note
    Backends **may** implement this for operators and types with a native reduction (e.g.
    `+` on `Float32`), dispatching on `typeof(op)`. The result has to be the one documented
    above, with Julia's semantics of `op` (e.g. NaN propagation and the sign of zero for `min`
    and `max`, wrap-around for integers), with one exception: for `+` and `*` on floating-point
    values, it may differ as if the values were combined in another order and grouping. The
    fallback combines ranges of doubling length with a butterfly of [`shfl_xor`](@ref), and
    in a partial sub-group broadcasts the result of the first lane with [`shfl`](@ref).
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
values of lanes `1` to `i`, as with [`sub_group_reduce`](@ref).

All work-items of the sub-group have to execute `sub_group_scan` together (not in a divergent
branch), with the same `op`; see [`shfl`](@ref). Values of the types that the shuffles
support are supported, see [`supports_shuffle`](@ref), and others for which the backend has a
native scan.

It exchanges values, not memory: it is not a memory fence, see [`sub_group_barrier`](@ref).

!!! note
    Backends **may** implement this for operators and types with a native scan (e.g. `+` on
    `Float32`), dispatching on `typeof(op)`, with the results documented above, as for
    [`sub_group_reduce`](@ref): for floating-point `+` and `*`, the values of each lane's
    prefix may be combined in another order and grouping. The fallback is a Hillis-Steele scan
    with [`shfl_up`](@ref).
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

"""
    sub_group_exclusive_scan(op, val::T, init::T)::T

The exclusive scan of `val` over the work-items of the sub-group with the associative binary
operator `op`, in the order of the lanes, starting from `init`: the work-item in lane 1 gets
`init` itself, and the one in lane `i > 1` gets `op(init, r)` for the reduction `r` of the
values of lanes `1` to `i - 1`, as with [`sub_group_scan`](@ref). `init` has to be the same
for all work-items of the sub-group, and of the type of `val`; it needn't be an identity of
`op`.

All work-items of the sub-group have to execute `sub_group_exclusive_scan` together (not in a
divergent branch), with the same `op`; see [`shfl`](@ref). Values of the types that the
shuffles support are supported, see [`supports_shuffle`](@ref), and others for which the
backend has a native scan.

It exchanges values, not memory: it is not a memory fence, see [`sub_group_barrier`](@ref).

!!! note
    Backends **may** implement this like [`sub_group_scan`](@ref). The fallback shifts the
    result of [`sub_group_scan`](@ref) up by a lane with [`shfl_up`](@ref).
"""
@inline function sub_group_exclusive_scan(op, val, init)
    prefix = shfl_up(sub_group_scan(op, val), 1)
    return get_sub_group_local_id(Int32) == Int32(1) ? init : op(init, prefix)
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

All work-items of a sub-group have to reach the same `sub_group_barrier()`; see
[`shfl`](@ref).

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
