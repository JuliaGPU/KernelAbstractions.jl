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
function __kernel(expr, generate_cpu = true, force_inbounds = false, unsafe_indices = false)
    def = splitdef(expr)
    name = def[:name]
    args = def[:args]
    generate_cpu && find_return(expr) && error(
        "Return statement not permitted in a kernel function $name",
    )

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
    cpu_name = Symbol(:cpu_, name)
    if generate_cpu
        def_cpu = deepcopy(def)
        def_cpu[:name] = cpu_name
        transform_cpu!(def_cpu, constargs, force_inbounds)
        cpu_function = combinedef(def_cpu)
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
                if $isgpu(dev)
                    return $construct(dev, sz, range, $gpu_name)
                else
                    if $generate_cpu
                        return $construct(dev, sz, range, $cpu_name)
                    else
                        error("This kernel is unavailable for backend CPU")
                    end
                end
            end
        end
    end

    if generate_cpu
        return Expr(:block, esc(cpu_function), esc(gpu_function), esc(constructors))
    else
        return Expr(:block, esc(gpu_function), esc(constructors))
    end
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
        append!(new_stmts, split(emit_gpu, body.args))
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

# The hard case, transform the function for CPU execution
# - mark constant arguments by applying `constify`.
# - insert aliasscope markers
# - insert implied loop bodys
#   - handle indices
#   - hoist workgroup definitions
#   - hoist uniform variables
function transform_cpu!(def, constargs, force_inbounds)
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
    if force_inbounds
        push!(new_stmts, Expr(:inbounds, true))
    end
    append!(new_stmts, split(emit_cpu, body.args))
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
    indices::Vector{Any}
    stmts::Vector{Any}
    allocations::Vector{Any}
    private_allocations::Vector{Any}
    private::Set{Symbol}
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
function split(
        emit,
        stmts,
        indices = Any[], private = Set{Symbol}(),
    )
    # 1. Split the code into blocks separated by `@synchronize`
    # 2. Aggregate `@index` expressions
    # 3. Hoist allocations
    # 4. Hoist uniforms

    current = Any[]
    allocations = Any[]
    private_allocations = Any[]
    new_stmts = Any[]
    for stmt in stmts
        has_sync = find_sync(stmt)
        if has_sync
            loop = WorkgroupLoop(deepcopy(indices), current, allocations, private_allocations, deepcopy(private), is_sync(stmt))
            push!(new_stmts, emit(loop))
            allocations = Any[]
            private_allocations = Any[]
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
                    new_args = unblock(split(emit, expr.args, deepcopy(indices), deepcopy(private)))
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
            push!(private, lhs)
            push!(private_allocations, :($lhs = $rhs))
            continue
        elseif @capture(stmt, lhs_ = rhs_ | (vs__, lhs_ = rhs_))
            if @capture(rhs, @index(args__))
                push!(indices, stmt)
                continue
            elseif @capture(rhs, @localmem(args__) | @uniform(args__))
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
                push!(private, lhs)
                continue
            end
        end

        push!(current, stmt)
    end

    # everything since the last `@synchronize`
    if !isempty(current)
        loop = WorkgroupLoop(deepcopy(indices), current, allocations, private_allocations, deepcopy(private), false)
        push!(new_stmts, emit(loop))
    end
    return new_stmts
end

function emit_cpu(loop)
    idx = gensym(:I)
    for stmt in loop.indices
        # splice index into the i = @index(Cartesian, $idx)
        @assert stmt.head === :(=)
        rhs = stmt.args[2]
        push!(rhs.args, idx)
    end
    stmts = Any[]
    append!(stmts, loop.allocations)

    # private_allocations turn into lhs = ntuple(i->rhs, length(__workitems_iterspace()))
    N = gensym(:N)
    push!(stmts, :($N = length($__workitems_iterspace(__ctx__))))

    for stmt in loop.private_allocations
        if @capture(stmt, lhs_ = rhs_)
            push!(stmts, :($lhs = ntuple(_ -> $rhs, $N)))
        else
            error("@private $stmt not an assignment")
        end
    end

    # don't emit empty loops
    if !(isempty(loop.stmts) || all(s -> s isa LineNumberNode, loop.stmts))
        body = Expr(:block, loop.stmts...)
        body = postwalk(body) do expr
            if @capture(expr, lhs_ = rhs_)
                if lhs in loop.private
                    error("Can't assign to variables marked private")
                end
            elseif @capture(expr, A_[i__])
                if A in loop.private
                    return :($A[$__index_Local_Linear(__ctx__, $(idx))][$(i...)])
                end
            elseif expr isa Symbol
                if expr in loop.private
                    return :($expr[$__index_Local_Linear(__ctx__, $(idx))])
                end
            end
            return expr
        end
        loopexpr = quote
            for $idx in $__workitems_iterspace(__ctx__)
                $__validindex(__ctx__, $idx) || continue
                $(loop.indices...)
                $(unblock(body))
            end
        end
        push!(stmts, loopexpr)
    end

    return unblock(Expr(:block, stmts...))
end

function emit_gpu(loop)
    stmts = Any[]

    body = Expr(:block, loop.stmts...)
    loopexpr = quote
        $(loop.allocations...)
        $(loop.private_allocations...)
        if __active_lane__
            $(loop.indices...)
            $(unblock(body))
        end
    end
    push!(stmts, loopexpr)
    if loop.terminated_in_sync
        push!(stmts, :($__synchronize()))
    end

    return unblock(Expr(:block, stmts...))
end
