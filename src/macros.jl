import MacroTools: splitdef, combinedef, isexpr, postwalk

function find_return(stmt)
    result = false
    postwalk(stmt) do expr
        result |= @capture(expr, return x_)
        expr
    end
    return result
end

# `quote` blocks insert `LineNumberNode`s pointing into this file. Rewriting them
# to the `@kernel` call site keeps coverage and profiling pointed at the user's
# code instead of at KernelAbstractions internals.
relocate_lines(expr, source::LineNumberNode) =
    postwalk(x -> x isa LineNumberNode ? source : x, expr)

# `MacroTools.unblock` drops `LineNumberNode`s when it collapses a block down to
# its single remaining statement. Only unwrap blocks that carry no line
# information, so that we never discard it.
function unblock_lines(ex)
    isexpr(ex, :block) || return ex
    length(ex.args) == 1 || return ex
    return unblock_lines(ex.args[1])
end

# XXX: Proper errors
function __kernel(expr, __source__::LineNumberNode, force_inbounds = false, unsafe_indices = false)
    def = splitdef(expr)
    name = def[:name]
    args = def[:args]

    constargs = Array{Bool}(undef, length(args))
    for (i, arg) in enumerate(args)
        if isexpr(arg, :macrocall)
            if arg.args[1] === Symbol("@Const")
                # arg.args[2] is a LineInfo node
                args[i] = arg.args[3] # strip @Const
                constargs[i] = true
                continue
            end
        end
        constargs[i] = false
    end

    def_gpu = deepcopy(def)
    def_gpu[:name] = gpu_name = Symbol(:gpu_, name)
    transform_gpu!(def_gpu, constargs, force_inbounds, unsafe_indices)
    gpu_function = combinedef(def_gpu)

    # create constructor functions
    _name = Symbol(:_, name)
    constructors = quote
        if $(name isa Symbol ? :(!@isdefined($name)) : true)
            function $_name(dev::Dev, sz::S, range::NDRange) where {Dev, S <: $_Size, NDRange <: $_Size}
                return $construct(dev, sz, range, $gpu_name)
            end
            Core.@__doc__ $name(dev) = $_name(dev, $DynamicSize(), $DynamicSize())
            $name(dev, size) = $_name(dev, $StaticSize(size), $DynamicSize())
            $name(dev, size, range) = $_name(dev, $StaticSize(size), $StaticSize(range))
            $name(dev, size::$_Size, range::$_Size) = $_name(dev, size, range)
        end
    end
    constructors = relocate_lines(constructors, __source__)

    return Expr(:block, esc(gpu_function), esc(constructors))
end

# The easy case, transform the function for GPU execution
# - mark constant arguments by applying `constify`.
function transform_gpu!(def, constargs, force_inbounds, unsafe_indices)
    let_constargs = Expr[]
    for (i, arg) in enumerate(def[:args])
        if constargs[i]
            push!(let_constargs, :($arg = $constify($arg)))
        end
    end
    pushfirst!(def[:args], :__ctx__)
    # `Any[]`, since `split` hands back `LineNumberNode`s alongside `Expr`s
    new_stmts = Any[]
    body = MacroTools.flatten(def[:body])
    push!(new_stmts, Expr(:aliasscope))
    if !unsafe_indices
        push!(new_stmts, :(__active_lane__ = $__validindex(__ctx__)))
    end
    if force_inbounds
        push!(new_stmts, Expr(:inbounds, true))
    end
    if !unsafe_indices
        append!(new_stmts, split(body.args))
    else
        push!(new_stmts, body)
    end
    if force_inbounds
        push!(new_stmts, Expr(:inbounds, :pop))
    end
    push!(new_stmts, Expr(:popaliasscope))
    push!(new_stmts, :(return nothing))
    def[:body] = Expr(
        :let,
        Expr(:block, let_constargs...),
        Expr(:block, new_stmts...),
    )
    return
end

struct WorkgroupLoop
    stmts::Vector{Any}
    allocations::Vector{Any}
    terminated_in_sync::Bool
    sync_line::Union{Nothing, LineNumberNode}
end

is_sync(expr) = @capture(expr, @synchronize() | @synchronize(a_))

function is_scope_construct(expr::Expr)
    return expr.head === :block # ||
    # expr.head === :let
end

function find_sync(stmt)
    result = false
    postwalk(stmt) do expr
        result |= is_sync(expr)
        expr
    end
    return result
end

function split(stmts)
    # 1. Split the code into blocks separated by `@synchronize`

    current = Any[]
    allocations = Any[]
    new_stmts = Any[]
    # `LineNumberNode` belonging to the statement currently being processed.
    # Statements are moved between `current` and `allocations` and the two end
    # up in different scopes of the emitted code, so instead of copying the line
    # information over eagerly we attach it to whichever list the statement
    # lands in. Otherwise hoisted allocations lose their source location.
    line = nothing
    # Flush the pending `LineNumberNode` into `stmts`.
    function take_line!(stmts)
        line === nothing && return
        push!(stmts, line)
        line = nothing
        return
    end

    for stmt in stmts
        if stmt isa LineNumberNode
            line = stmt
            continue
        end

        has_sync = find_sync(stmt)
        if has_sync
            loop = WorkgroupLoop(current, allocations, is_sync(stmt), line)
            push!(new_stmts, emit(loop))
            allocations = Any[]
            current = Any[]
            if is_sync(stmt)
                # `emit` consumed `line` for the `@synchronize` itself
                line = nothing
                continue
            end

            # Recurse into scope constructs
            # TODO: This currently implements hard scoping
            #       probably need to implemet soft scoping
            #       by not deepcopying the environment.
            recurse(x) = x
            function recurse(expr::Expr)
                expr = unblock_lines(expr)
                if is_scope_construct(expr) && any(find_sync, expr.args)
                    return Expr(expr.head, split(expr.args)...)
                else
                    return Expr(expr.head, map(recurse, expr.args)...)
                end
            end
            take_line!(new_stmts)
            push!(new_stmts, recurse(stmt))
            continue
        end

        if @capture(stmt, @uniform x_)
            take_line!(allocations)
            push!(allocations, stmt)
            continue
        elseif @capture(stmt, @private lhs_ = rhs_)
            take_line!(allocations)
            push!(allocations, :($lhs = $rhs))
            continue
        elseif @capture(stmt, lhs_ = rhs_ | (vs__, lhs_ = rhs_))
            if @capture(rhs, @localmem(args__) | @uniform(args__))
                take_line!(allocations)
                push!(allocations, stmt)
                continue
            elseif @capture(rhs, @private(T_, dims_))
                # Implement the legacy `mem = @private T dims` as
                # mem = Scratchpad(T, Val(dims))

                if dims isa Integer
                    dims = (dims,)
                end
                alloc = :($Scratchpad(__ctx__, $T, Val($dims)))
                take_line!(allocations)
                push!(allocations, :($lhs = $alloc))
                continue
            end
        end

        take_line!(current)
        push!(current, stmt)
    end

    # everything since the last `@synchronize`
    if !isempty(current)
        loop = WorkgroupLoop(current, allocations, false, nothing)
        push!(new_stmts, emit(loop))
    end
    return new_stmts
end

function emit(loop)
    # Note: built without `quote`, since that would splice `LineNumberNode`s
    # pointing at this file into the middle of the user's kernel body.
    stmts = Any[]

    append!(stmts, loop.allocations)
    push!(stmts, Expr(:if, :__active_lane__, Expr(:block, loop.stmts...)))
    if loop.terminated_in_sync
        loop.sync_line === nothing || push!(stmts, loop.sync_line)
        push!(stmts, :($__synchronize()))
    end

    return Expr(:block, stmts...)
end
