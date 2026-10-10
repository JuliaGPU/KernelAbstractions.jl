###
# Work-group reductions and scans
# - @groupreduce
# - @groupscan
###

export @groupreduce, @groupscan

"""
    @groupreduce(op, val, neutral; groupsize)

Reduce `val` over all work-items of the workgroup with the binary operator `op`, and return
the result on every work-item. `op` has to be associative and commutative: the values are
combined in an unspecified order. `neutral` has to be its neutral element
(`op(neutral, x) == op(x, neutral) == x`). For example, to find the smallest value and its
index, reduce `(value, index)` pairs with an operator that breaks ties by the index, such
as `min` on tuples, with the neutral element `(typemax(T), typemax(Int))`.

The result has the type of `neutral`, which `val` is converted to: `op(x, y)` has to return
that type `T = typeof(neutral)` for arguments of type `T`.

`@groupreduce` is a collective, like [`@synchronize`](@ref): it has to be used as a
statement on its own (`res = @groupreduce(op, val, neutral)`), and reached by all
work-items of the workgroup, not in a branch or loop that only some of them execute, or
after an early `return`. Work-items that pad a partial workgroup (outside of the `ndrange`)
take part as well: they contribute `neutral`, without evaluating `val`. So `op` and
`neutral` have to be the same for the whole workgroup, and computable on those work-items.

The reduction stores values in local memory, so its size has to be known at compile time:
either the kernel has a static workgroup size, or the keyword `groupsize` gives an upper
bound of the workgroup size (a constant, e.g. a literal or a type parameter of the kernel).

```julia
@kernel function sum_kernel!(out, @Const(x))
    i = @index(Global)
    res = @groupreduce(+, x[i], zero(eltype(out)); groupsize = 1024)
    if @index(Local, Linear) == 1
        out[@index(Group, Linear)] = res
    end
end
```
"""
macro groupreduce(args...)
    op, val, neutral, options = parse_collective("@groupreduce", args, (:groupsize,))
    bound = collective_bound(Base.get(options, :groupsize, nothing))
    return __collective_call(neutral, val) do neutral, val
        :($__groupreduce($(esc(:__ctx__)), $(esc(op)), $val, $neutral, $bound))
    end
end

"""
    @groupscan(op, val, neutral; groupsize, inclusive = true)

Scan `val` over the work-items of the workgroup with the binary operator `op`, in the order
of `@index(Local, Linear)`: the work-item with local index `i` gets
`op(...op(op(val₁, val₂), val₃)..., valᵢ)` (inclusive), or the same up to `valᵢ₋₁` and
`neutral` for the first work-item (with `inclusive = false`). `op` has to be associative,
but needn't be commutative, and `neutral` its neutral element
(`op(neutral, x) == op(x, neutral) == x`). The result has the type of `neutral`, which `val`
is converted to, and `op` has to return that type.

Like [`@groupreduce`](@ref), `@groupscan` is a collective that has to be used as a statement
on its own and reached by all work-items of the workgroup; padding work-items contribute
`neutral`. The scan stores two values per work-item in local memory, sized like for
`@groupreduce`: either the kernel has a static workgroup size, or the keyword `groupsize`
gives an upper bound of the workgroup size. `inclusive` has to be a constant as well.

For example, to compact the elements of `x` that satisfy `pred` within each workgroup:

```julia
@kernel function compact!(out, counts, @Const(x), pred)
    i = @index(Global, Linear)
    keep = pred(x[i])
    offset = @groupscan(+, Int32(keep), Int32(0); inclusive = false)
    total = @groupreduce(+, Int32(keep), Int32(0))
    base = (@index(Group, Linear) - 1) * prod(@groupsize())
    if keep
        out[base + offset + 1] = x[i]
    end
    if @index(Local, Linear) == 1
        counts[@index(Group, Linear)] = total
    end
end
```
"""
macro groupscan(args...)
    op, val, neutral, options = parse_collective("@groupscan", args, (:groupsize, :inclusive))
    bound = collective_bound(Base.get(options, :groupsize, nothing))
    inclusive = Base.get(options, :inclusive, true)
    return __collective_call(neutral, val) do neutral, val
        :($__groupscan($(esc(:__ctx__)), $(esc(op)), $val, $neutral, $bound, Val($(esc(inclusive)))))
    end
end

collective_bound(::Nothing) = :($__static_groupsize($(esc(:__ctx__))))
collective_bound(groupsize) = :(Val($(esc(groupsize))))

# Convert `val` to the type of `neutral` *before* the call of the collective. The type of
# `val` may differ between the work-items, e.g. `Union{Float32, Float64}` for an accumulator
# that only some work-items added a `Float64` to, or because padding work-items contribute
# `neutral` instead of `val` (see `mask_collective`). Julia union-splits a call with such an
# argument into one call per type, so the work-items would execute different copies of the
# collective, and its barriers and shuffles. Only the `convert` may be split this way.
function __collective_call(f, neutral, val)
    n, v = gensym(:neutral), gensym(:val)
    return quote
        let $v = $(esc(val)), $n = $(esc(neutral))
            $(f(n, :($convert($typeof($n), $v))))
        end
    end
end

# The `op`, `val` and `neutral` of a collective's arguments, and its `key = value` options
# (also after a `;`), which have to be among `keys`.
function parse_collective(name, args, keys)
    positional = Any[]
    options = Dict{Symbol, Any}()
    function option!(ex)
        key = ex.args[1]
        (key isa Symbol && key in keys) ||
            error("$name: unknown option `$key`, expected one of $(join(keys, ", "))")
        haskey(options, key) && error("$name: option `$key` given more than once")
        options[key] = ex.args[2]
        return
    end
    for arg in args
        if isexpr(arg, :parameters)
            foreach(option!, arg.args)
        elseif isexpr(arg, :(=)) || isexpr(arg, :kw)
            option!(arg)
        else
            push!(positional, arg)
        end
    end
    length(positional) == 3 ||
        error("$name expects `op`, `val` and `neutral`, and the options $(join(keys, ", ")) as keywords")
    return positional..., options
end

const COLLECTIVES = (Symbol("@groupreduce"), Symbol("@groupscan"))

# Whether `expr` is a collective that all work-items of a workgroup take part in.
is_collective(expr) = any(name -> is_macrocall(expr, name), COLLECTIVES)

# A collective that is used as a statement: `@groupreduce(...)` or `lhs = @groupreduce(...)`.
function is_collective_stmt(stmt)
    is_collective(stmt) && return true
    isexpr(stmt, :(=)) && is_collective(stmt.args[2]) || return false
    lhs = stmt.args[1]
    lhs isa Symbol && return true
    isexpr(lhs, :(::)) && lhs.args[1] isa Symbol && return true
    isexpr(lhs, :tuple) && all(x -> x isa Symbol, lhs.args) && return true
    return false
end

collective_error(expr) = error(
    "`$(expr.args[1])` must be used as a statement of its own, " *
        "e.g. `res = $(expr.args[1])(op, val, neutral)`, found `$(expr)`"
)

# Rewrite a collective statement, so that padding work-items contribute the neutral element
# instead of evaluating the value. `neutral` is evaluated once, before the collective.
function mask_collective(stmt)
    if isexpr(stmt, :(=))
        binding, call = mask_collective(stmt.args[2]).args
        return Expr(:block, binding, Expr(:(=), stmt.args[1], call))
    end
    # `args[2]` is the macro's `LineNumberNode`, options may come first after a `;`
    args = copy(stmt.args)
    i = 3
    while i <= length(args) && (isexpr(args[i], :parameters) || args[i] isa LineNumberNode)
        i += 1
    end
    length(args) >= i + 2 || error("`$(args[1])` expects `op`, `val` and `neutral`")
    val, neutral = args[i + 1], args[i + 2]
    n = gensym(:neutral)
    args[i + 1] = :(__active_lane__ ? $val : $n)
    args[i + 2] = n
    return Expr(:block, :($n = $neutral), Expr(:macrocall, args...))
end

# The workgroup size as a `Val`, if it is static.
@inline __static_groupsize(ctx::CompilerMetadata) = __static_groupsize(__iterspace(ctx))
@inline __static_groupsize(::NDRange{N, B, W}) where {N, B, W <: StaticSize} = Val(prod(get(W)))
@inline __static_groupsize(::NDRange) = throw(
    ArgumentError(
        "group reductions and scans require a static workgroup size or an upper bound of it"
    )
)

# The largest power of two smaller than `n`, or 0.
@inline function __prevpow2(n::T) where {T <: Integer}
    n <= one(T) && return zero(T)
    return one(T) << (8 * sizeof(T) - 1 - leading_zeros(n - one(T)))
end

@inline function __groupreduce(ctx, op, val, neutral::T, ::Val{N}) where {T, N}
    n = prod(groupsize(ctx))
    n <= N || throw(ArgumentError("@groupreduce: the workgroup size exceeds the given upper bound"))
    storage = KI.localmemory(T, Val(N))
    res = __groupreduce_tree(op, convert(T, val), storage, __index_Local_Linear(ctx), n)
    # all work-items have to read the result before the storage can be reused
    KI.barrier()
    return res
end

# Tree reduction in local memory, folding the upper half of the values onto the lower half.
@inline function __groupreduce_tree(op, val, storage, lid, n)
    @inbounds storage[lid] = val
    KI.barrier()
    s = __prevpow2(n)
    while s > 0
        if lid <= s && lid + s <= n
            @inbounds storage[lid] = op(storage[lid], storage[lid + s])
        end
        KI.barrier()
        s >>= 1
    end
    return @inbounds storage[1]
end

@inline function __groupscan(ctx, op, val, neutral::T, ::Val{N}, ::Val{inclusive}) where {T, N, inclusive}
    n = prod(groupsize(ctx))
    n <= N || throw(ArgumentError("@groupscan: the workgroup size exceeds the given upper bound"))
    # two buffers, the scan reads from one and writes to the other
    storage = KI.localmemory(T, Val(2 * N))
    lid = __index_Local_Linear(ctx)

    # Hillis-Steele: after the step with distance `d`, every work-item holds the scan of the
    # (up to) `2d` values ending at its own
    src = 0
    @inbounds storage[lid] = convert(T, val)
    KI.barrier()
    d = 1
    while d < n
        x = @inbounds storage[src + lid]
        if lid > d
            x = op(@inbounds(storage[src + lid - d]), x)
        end
        @inbounds storage[(N - src) + lid] = x
        KI.barrier()
        src = N - src
        d <<= 1
    end

    if inclusive
        res = @inbounds storage[src + lid]
    else
        res = lid == 1 ? neutral : @inbounds storage[src + lid - 1]
    end
    # all work-items have to read their result before the storage can be reused
    KI.barrier()
    return res
end

