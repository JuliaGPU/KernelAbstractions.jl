import MacroTools: splitdef, combinedef, isexpr, postwalk

function find_return(stmt)
    result = false
    postwalk(stmt) do expr
        result |= @capture(expr, return x_)
        expr
    end
    result
end

# XXX: Proper errors
function __kernel(expr)
    def = splitdef(expr)
    name = def[:name]
    args = def[:args]

    find_return(expr) && error("Return statement not permitted in a kernel function $name")

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

    # create two functions
    # 1. GPU function
    # 2. CPU function with work-group loops inserted
    #
    # Without the deepcopy we might accidentially modify expr shared between CPU and GPU
    def_cpu = deepcopy(def)
    def_gpu = deepcopy(def)

    def_cpu[:name] = cpu_name = Symbol(:cpu_, name)
    def_gpu[:name] = gpu_name = Symbol(:gpu_, name)

    transform_cpu!(def_cpu, constargs)
    transform_gpu!(def_gpu, constargs)

    cpu_function = combinedef(def_cpu)
    gpu_function = combinedef(def_gpu)

    # create constructor functions
    constructors = quote
        if $(name isa Symbol ? :(!@isdefined($name)) : true)
            Core.@__doc__ $name(dev::$Device) = $name(dev, $DynamicSize(), $DynamicSize())
            $name(dev::$Device, size) = $name(dev, $StaticSize(size), $DynamicSize())
            $name(dev::$Device, size, range) = $name(dev, $StaticSize(size), $StaticSize(range))
            function $name(::Device, ::S, ::NDRange) where {Device<:$CPU, S<:$_Size, NDRange<:$_Size}
                return $Kernel{Device, S, NDRange, typeof($cpu_name)}($cpu_name)
            end
            function $name(::Device, ::S, ::NDRange) where {Device<:$GPU, S<:$_Size, NDRange<:$_Size}
                return $Kernel{Device, S, NDRange, typeof($gpu_name)}($gpu_name)
            end
        end
    end

    return Expr(:block, esc(cpu_function), esc(gpu_function), esc(constructors))
end

# The easy case, transform the function for GPU execution
# - mark constant arguments by applying `constify`.
function transform_gpu!(def, constargs)
    let_constargs = Expr[]
    for (i, arg) in enumerate(def[:args])
        if constargs[i]
            push!(let_constargs, :($arg = $constify($arg)))
        end
    end

    body = quote
        if $__validindex()
            $(def[:body])
        end
        return nothing
    end
    def[:body] = Expr(:let,
        Expr(:block, let_constargs...),
        body,
    )
end

# The hard case, transform the function for CPU execution
# - mark constant arguments by applying `constify`.
# - insert aliasscope markers
# - insert implied loop bodys
#   - handle indicies
#   - hoist workgroup definitions
#   - hoist uniform variables
function transform_cpu!(def, constargs)
    let_constargs = Expr[]
    for (i, arg) in enumerate(def[:args])
        if constargs[i]
            push!(let_constargs, :($arg = $constify($arg)))
        end
    end
    new_stmts = Expr[]
    body = MacroTools.flatten(def[:body])
    push!(new_stmts, Expr(:aliasscope))
    append!(new_stmts, split(body.args))
    push!(new_stmts, Expr(:popaliasscope))
    push!(new_stmts, :(return nothing))
    def[:body] = Expr(:let,
        Expr(:block, let_constargs...),
        Expr(:block, new_stmts...)
    )
end

struct WorkgroupLoop
    indicies :: Vector{Any}
    stmts :: Vector{Any}
    allocations :: Vector{Any}
    private :: Vector{Any}
end

is_sync(expr) = @capture(expr, @synchronize() | @synchronize(a_))

function is_scope_construct(expr::Expr)
    expr.head === :block # ||
    # expr.head === :let
end

function find_sync(stmt)
    result = false
    postwalk(stmt) do expr
        result |= is_sync(expr)
        expr
    end
    result
end

# TODO proper handling of LineInfo
function split(stmts,
               indicies = Any[], private=Any[])
    # 1. Split the code into blocks separated by `@synchronize`
    # 2. Aggregate `@index` expressions
    # 3. Hoist allocations
    # 4. Hoist uniforms

    current     = Any[]
    allocations = Any[]
    new_stmts   = Any[]
    for stmt in stmts
        has_sync = find_sync(stmt)
        if has_sync
            loop = WorkgroupLoop(deepcopy(indicies), current, allocations, deepcopy(private))
            push!(new_stmts, emit(loop))
            allocations = Any[]
            current     = Any[]
            is_sync(stmt) && continue

            # Recurse into scope constructs
            # TODO: This currently implements hard scoping
            #       probably need to implemet soft scoping
            #       by not deepcopying the environment.
            recurse(x) = x
            function recurse(expr::Expr)
                expr = unblock(expr)
                if is_scope_construct(expr) && any(find_sync, expr.args)
                    new_args = unblock(split(expr.args, deepcopy(indicies), deepcopy(private)))
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
        elseif @capture(stmt, lhs_ = rhs_ | (vs__, lhs_ = rhs_))
            if @capture(rhs, @index(args__))
                push!(indicies, stmt)
                continue
            elseif @capture(rhs, @localmem(args__) | @uniform(args__) )
                push!(allocations, stmt)
                continue
            elseif @capture(rhs, @private(args__))
                push!(allocations, stmt)
                push!(private, lhs)
                continue
            end
        end

        push!(current, stmt)
    end

    # everything since the last `@synchronize`
    if !isempty(current)
        loop = WorkgroupLoop(deepcopy(indicies), current, allocations, deepcopy(private))
        push!(new_stmts, emit(loop))
    end
    return new_stmts
end

function emit(loop)
    idx = gensym(:I)
    for stmt in loop.indicies
        # splice index into the i = @index(Cartesian, $idx)
        @assert stmt.head === :(=)
        rhs = stmt.args[2]
        push!(rhs.args, idx)
    end
    stmts = Any[]
    append!(stmts, loop.allocations)
    # don't emit empty loops
    if !(isempty(loop.stmts) || all(s->s isa LineNumberNode, loop.stmts))
        body = Expr(:block, loop.stmts...)
        body = postwalk(body) do expr
            if @capture(expr, A_[i__])
                if A in loop.private
                    return :($A[$(i...), $(idx).I...])
                end
            end
            return expr
        end
        loopexpr = quote
            for $idx in $__workitems_iterspace()
                $__validindex($idx) || continue
                $(loop.indicies...)
                $(unblock(body))
            end
        end
        push!(stmts, loopexpr)
    end

    return unblock(Expr(:block, stmts...))
end
