###
# Group and sub-group reductions
# - @groupreduce
# - @subgroupreduce
###

export @groupreduce, @subgroupreduce

"""
    @groupreduce(op, val, neutral[, groupsize]; subgroups = false)

Reduce `val` over all work-items of the workgroup with the binary operator `op`, and return
the result on every work-item. `op` has to be associative and commutative, and `neutral`
its neutral element (`op(neutral, x) == x`). The result has the type of `neutral`, which
`val` is converted to.

Work-items that are not part of the `ndrange` (padding of a partial workgroup) contribute
`neutral`, and don't evaluate `val`. Like [`@synchronize`](@ref), `@groupreduce` must be
reached by all work-items of the workgroup, and has to be used as a statement on its own:
`res = @groupreduce(op, val, neutral)`.

The reduction uses local memory for one value per work-item, so its size has to be known at
compile time: either the kernel has a static workgroup size, or `groupsize` gives an upper
bound of the workgroup size (a constant, e.g. a literal or a type parameter of the kernel).

With `subgroups = true` each sub-group first reduces its values with
[`KernelInterface.shfl_down`](@ref), which needs fewer barriers. It must only be used on
backends that support shuffles of the type of `neutral`, see
[`KernelInterface.supports_shuffle`](@ref). `subgroups` has to be a constant as well, e.g.
a type parameter set from the host:

```julia
@kernel function sum_kernel!(out, @Const(x), ::Val{S}) where {S}
    i = @index(Global)
    res = @groupreduce(+, x[i], zero(eltype(out)), 1024; subgroups = S)
    if @index(Local, Linear) == 1
        out[@index(Group, Linear)] = res
    end
end

subgroups = KernelInterface.supports_shuffle(backend, eltype(out))
sum_kernel!(backend)(out, x, Val(subgroups); ndrange = length(x))
```
"""
macro groupreduce(args...)
    positional, options = split_options(args, (:groupsize, :subgroups))
    3 <= length(positional) <= 4 ||
        error("@groupreduce expects `op`, `val`, `neutral` and optionally `groupsize`")
    op, val, neutral = positional
    groupsize = length(positional) == 4 ? positional[4] : Base.get(options, :groupsize, nothing)
    bound = groupsize === nothing ? :($__static_groupsize($(esc(:__ctx__)))) :
        :(Val($(esc(groupsize))))
    subgroups = Base.get(options, :subgroups, false)
    return quote
        $__groupreduce(
            $(esc(:__ctx__)), $(esc(op)), $(esc(val)), $(esc(neutral)),
            $bound, Val($(esc(subgroups))),
        )
    end
end

"""
    @subgroupreduce(op, val, neutral)

Reduce `val` over the work-items of the sub-group with the binary operator `op`, using
[`KernelInterface.shfl_down`](@ref). The result is only defined on the first work-item of
the sub-group (`KernelInterface.get_sub_group_local_id() == 1`). `op` has to be associative,
and `neutral` its neutral element. The result has the type of `neutral`.

Work-items that are not part of the `ndrange` contribute `neutral`. `@subgroupreduce` must be
reached by all work-items of the sub-group, and has to be used as a statement on its own:
`res = @subgroupreduce(op, val, neutral)`.

It must only be used on backends that support shuffles of the type of `neutral`, see
[`KernelInterface.supports_shuffle`](@ref).
"""
macro subgroupreduce(op, val, neutral)
    return :($__subgroupreduce($(esc(op)), $(esc(val)), $(esc(neutral))))
end

# Separate `key = value` options (also after a `;`) from the positional macro arguments.
function split_options(args, keys)
    positional = Any[]
    options = Dict{Symbol, Any}()
    function option!(ex)
        (ex.args[1] isa Symbol && ex.args[1] in keys) ||
            error("unknown option `$(ex.args[1])`, expected one of $(join(keys, ", "))")
        options[ex.args[1]] = ex.args[2]
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
    return positional, options
end

const COLLECTIVES = (Symbol("@groupreduce"), Symbol("@subgroupreduce"))

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
# instead of evaluating the value.
function mask_collective(stmt)
    if isexpr(stmt, :(=))
        return Expr(:(=), stmt.args[1], mask_collective(stmt.args[2]))
    end
    # `args[2]` is the macro's `LineNumberNode`, options may come first after a `;`
    args = copy(stmt.args)
    i = 3
    while i <= length(args) && (isexpr(args[i], :parameters) || args[i] isa LineNumberNode)
        i += 1
    end
    length(args) >= i + 2 || error("`$(args[1])` expects at least `op`, `val` and `neutral`")
    val, neutral = args[i + 1], args[i + 2]
    args[i + 1] = :(__active_lane__ ? $val : $neutral)
    return Expr(:macrocall, args...)
end

# The workgroup size as a `Val`, if it is static.
@inline __static_groupsize(ctx::CompilerMetadata) = __static_groupsize(__iterspace(ctx))
@inline __static_groupsize(::NDRange{N, B, W}) where {N, B, W <: StaticSize} = Val(prod(get(W)))
@inline __static_groupsize(::NDRange) = throw(
    ArgumentError(
        "@groupreduce requires a static workgroup size or an upper bound of it"
    )
)

# The largest power of two smaller than `n`, or 0.
@inline function __prevpow2(n::T) where {T <: Integer}
    n <= one(T) && return zero(T)
    return one(T) << (8 * sizeof(T) - 1 - leading_zeros(n - one(T)))
end

@inline function __groupreduce(ctx, op, val, neutral::T, ::Val{N}, ::Val{subgroups}) where {T, N, subgroups}
    n = prod(groupsize(ctx))
    n <= N || throw(ArgumentError("@groupreduce: the workgroup size exceeds the given upper bound"))
    storage = KI.localmemory(T, Val(N))
    lid = __index_Local_Linear(ctx)
    if subgroups
        res = __groupreduce_subgroups(op, convert(T, val), neutral, storage)
    else
        res = __groupreduce_tree(op, convert(T, val), storage, lid, n)
    end
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

# Reduce every sub-group with shuffles, then reduce the results of the sub-groups in the first one.
@inline function __groupreduce_subgroups(op, val, neutral, storage)
    sg = KI.get_sub_group_id()
    lane = KI.get_sub_group_local_id()
    val = __subgroupreduce(op, val, neutral)
    if lane == 1
        @inbounds storage[sg] = val
    end
    KI.barrier()

    # every sub-group reduces the results of all sub-groups, which avoids running shuffles
    # in a branch that only some sub-groups take
    width = KI.get_sub_group_size()
    acc = neutral
    i = lane
    while i <= KI.get_num_sub_groups()
        acc = op(acc, @inbounds storage[i])
        i += width
    end
    acc = __subgroupreduce(op, acc, neutral)
    KI.barrier()
    if sg == 1 && lane == 1
        @inbounds storage[1] = acc
    end
    KI.barrier()
    return @inbounds storage[1]
end

# Combine contiguous ranges of lanes of doubling length, so that the first lane ends up with
# the reduction of the sub-group. Only requires `op` to be associative.
@inline function __subgroupreduce(op, val, neutral::T) where {T}
    val = convert(T, val)
    lane = KI.get_sub_group_local_id()
    sgsize = KI.get_sub_group_size()
    offset = 1
    while offset < sgsize
        other = KI.shfl_down(val, offset)
        # the result of shuffling from past the end of the sub-group is unspecified
        if lane + offset <= sgsize
            val = op(val, other)
        end
        offset <<= 1
    end
    return val
end
