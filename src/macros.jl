import MacroTools: splitdef, combinedef, isexpr, postwalk

function find_return(stmt)
    result = false
    postwalk(stmt) do expr
        result |= @capture(expr, return x_)
        expr
    end
    return result
end

# XXX: Proper errors
function __kernel(expr, force_inbounds = false, unsafe_indices = false)
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
    constructors = quote
        if $(name isa Symbol ? :(!@isdefined($name)) : true)
            Core.@__doc__ $name(dev) = $name(dev, $DynamicSize(), $DynamicSize())
            $name(dev, size) = $name(dev, $StaticSize(size), $DynamicSize())
            $name(dev, size, range) = $name(dev, $StaticSize(size), $StaticSize(range))
            function $name(dev::Dev, sz::S, range::NDRange) where {Dev, S <: $_Size, NDRange <: $_Size}
                return $construct(dev, sz, range, $gpu_name)
            end
        end
    end

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
    new_stmts = Expr[]
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

# TODO proper handling of LineInfo
function split(stmts)
    # 1. Split the code into blocks separated by `@synchronize`

    current = Any[]
    allocations = Any[]
    new_stmts = Any[]
    for stmt in stmts
        has_sync = find_sync(stmt)
        if has_sync
            loop = WorkgroupLoop(current, allocations, is_sync(stmt))
            push!(new_stmts, emit(loop))
            allocations = Any[]
            current = Any[]
            is_sync(stmt) && continue

            # Recurse into scope constructs
            # TODO: This currently implements hard scoping
            #       probably need to implemet soft scoping
            #       by not deepcopying the environment.
            recurse(x) = x
            function recurse(expr::Expr)
                expr = unblock(expr)
                if is_scope_construct(expr) && any(find_sync, expr.args)
                    new_args = unblock(split(expr.args))
                    return Expr(expr.head, new_args...)
                else
                    return Expr(expr.head, map(recurse, expr.args)...)
                end
            end
            push!(new_stmts, recurse(stmt))
            continue
        end

        if @capture(stmt, @uniform x_)
            push!(allocations, stmt)
            continue
        elseif @capture(stmt, @private lhs_ = rhs_)
            push!(allocations, :($lhs = $rhs))
            continue
        elseif @capture(stmt, lhs_ = rhs_ | (vs__, lhs_ = rhs_))
            if @capture(rhs, @localmem(args__) | @uniform(args__))
                push!(allocations, stmt)
                continue
            elseif @capture(rhs, @private(T_, dims_))
                # Implement the legacy `mem = @private T dims` as
                # mem = Scratchpad(T, Val(dims))

                if dims isa Integer
                    dims = (dims,)
                end
                alloc = :($Scratchpad(__ctx__, $T, Val($dims)))
                push!(allocations, :($lhs = $alloc))
                continue
            end
        end

        push!(current, stmt)
    end

    # everything since the last `@synchronize`
    if !isempty(current)
        loop = WorkgroupLoop(current, allocations, false)
        push!(new_stmts, emit(loop))
    end
    return new_stmts
end

function emit(loop)
    stmts = Any[]

    body = Expr(:block, loop.stmts...)
    loopexpr = quote
        $(loop.allocations...)
        if __active_lane__
            $(unblock(body))
        end
    end
    push!(stmts, loopexpr)
    if loop.terminated_in_sync
        push!(stmts, :($__synchronize()))
    end

    return unblock(Expr(:block, stmts...))
end
