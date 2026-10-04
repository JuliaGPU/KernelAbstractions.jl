import MacroTools: splitdef, combinedef, isexpr, postwalk

function find_return(stmt)
    result = Ref(false)
    postwalk(stmt) do expr
        result[] |= @capture(expr, return x_)
        expr
    end
    return result[]
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
function __kernel(expr, __source__::LineNumberNode, __module__::Module, force_inbounds = false, unsafe_indices = false, generated = false)
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
    if generated
        # Turn the kernel into a generated function: the transformed body is
        # quoted so that it is returned as an expression. Passing the quote
        # through `macroexpand` (one level only, we do not want to expand the
        # macros *inside* the quoted body here) lowers the `$` interpolations
        # into the code that builds the expression at generation time.
        body = macroexpand(__module__, Expr(:quote, def_gpu[:body]), recursive = false)
        # Inference swallows any error thrown while generating (the kernel then
        # merely infers to `Any`, which GPUCompiler reports as "kernel returns a
        # value of type `Any`" without ever showing the cause), and there is no
        # common launch path across backends where we could rethrow it. So
        # catch the error here and hand it back through the return type
        # instead, where GPUCompiler's validation prints it on every backend.
        body = quote
            try
                $(check_generated)($(__module__), $body)
            catch err
                $(generated_error_body)(err)
            end
        end
        def_gpu[:body] = Expr(:if, Expr(:generated), body, Expr(:meta, :generated_only))
    end
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

"""
    GeneratedKernelError{Msg}

Marker type a `generated=true` kernel returns when its generator failed, carrying the
error message in its type parameter. A kernel that returns this shows up in the
`KernelError` GPUCompiler raises for kernels that return a value, which is the only
channel through which a failure at generation time can be reported.
"""
struct GeneratedKernelError{Msg} end
GeneratedKernelError(msg::AbstractString) = GeneratedKernelError{Symbol(msg)}()

function Base.show(io::IO, ::Type{GeneratedKernelError{Msg}}) where {Msg}
    return print(io, "KernelAbstractions.GeneratedKernelError(", repr(String(Msg)), ")")
end

# The left-hand side of a short-form method definition, `f(x) = ...`, possibly
# wrapped in `where` clauses and a return type annotation: `f(x)::T where {T} = ...`.
function is_short_def_lhs(lhs)
    while isexpr(lhs, :where) || isexpr(lhs, :(::))
        isempty(lhs.args) && return false
        lhs = lhs.args[1]
    end
    return isexpr(lhs, :call)
end

# Runs inside the generator: makes sure the generated body is something Julia
# accepts as the result of a generated function. Julia itself only rejects a
# closure, comprehension or generator when lowering the body, which happens
# outside the generator's `try` and thus can't be turned into a useful error
# there, so check for them up front.
function check_generated(mod::Module, body)
    ex = macroexpand(mod, body)
    MacroTools.postwalk(ex) do node
        if isexpr(node, :->) || isexpr(node, :function) || isexpr(node, :do) ||
                isexpr(node, :comprehension) || isexpr(node, :generator) ||
                isexpr(node, :flatten) || (isexpr(node, :(=)) && is_short_def_lhs(node.args[1]))
            found = replace(string(MacroTools.striplines(node)), r"\s+" => " ")
            error(
                "the body of a `generated=true` kernel cannot contain a closure, " *
                    "comprehension or generator (found `", found, "`). " *
                    "Use `Base.Cartesian.@nexprs \$N` or `@ntuple \$N` instead.",
            )
        end
        return node
    end
    return body
end

# Runs inside the generator, so it must not use code reflection: `showerror` for a
# `MethodError` looks up candidate methods, which is forbidden there, so format
# that one by hand and fall back to the bare exception type for anything else
# that can't be shown.
function generated_error_message(err)
    if err isa MethodError
        sig = join((a isa Type ? "::Type{$a}" : "::$(typeof(a))" for a in err.args), ", ")
        return string("MethodError: no method matching ", err.f, "(", sig, ")")
    end
    msg = try
        sprint(showerror, err)
    catch
        string(typeof(err))
    end
    return first(Base.split(msg, '\n'))
end

function generated_error_body(err)
    msg = try
        generated_error_message(err)
    catch
        string(typeof(err))
    end
    # A `Symbol` cannot contain NUL characters.
    msg = replace(msg, '\0' => "\\0")
    return :(return $(GeneratedKernelError(msg)))
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

# Whether `expr` calls the kernel-language macro `name`. Also recognizes qualified uses,
# e.g. `KA.@synchronize()`, which would otherwise not be split or hoisted.
function is_macrocall(expr, name::Symbol)
    isexpr(expr, :macrocall) || return false
    callee = expr.args[1]
    if isexpr(callee, :.)
        callee = callee.args[end]
        callee isa QuoteNode && (callee = callee.value)
    elseif callee isa GlobalRef
        callee = callee.name
    end
    return callee === name
end

is_sync(expr) = is_macrocall(expr, Symbol("@synchronize"))

# Split the branches of an `if` that contains a `@synchronize`. The conditions run on all
# lanes, but every branch needs to be guarded, including those without a `@synchronize`.
function split_branches(expr, recurse)
    cond = expr.args[1]
    find_sync(cond) && (cond = recurse(cond))
    branches = map(expr.args[2:end]) do branch
        if isexpr(branch, :elseif)
            split_branches(branch, recurse)
        else
            stmts = isexpr(branch, :block) ? branch.args : Any[branch]
            Expr(:block, split(stmts)...)
        end
    end
    return Expr(expr.head, cond, branches...)
end

function is_scope_construct(expr::Expr)
    return expr.head === :block # ||
    # expr.head === :let
end

function find_sync(stmt)
    result = Ref(false)
    postwalk(stmt) do expr
        result[] |= is_sync(expr)
        expr
    end
    return result[]
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
            #       probably need to implement soft scoping
            #       by not deepcopying the environment.
            recurse(x) = x
            function recurse(expr::Expr)
                expr = unblock_lines(expr)
                if expr.head in (:if, :elseif) && find_sync(expr)
                    return split_branches(expr, recurse)
                elseif is_scope_construct(expr) && any(find_sync, expr.args)
                    return Expr(expr.head, split(expr.args)...)
                else
                    return Expr(expr.head, map(recurse, expr.args)...)
                end
            end
            take_line!(new_stmts)
            push!(new_stmts, recurse(stmt))
            continue
        end

        if is_macrocall(stmt, Symbol("@uniform"))
            take_line!(allocations)
            push!(allocations, stmt)
            continue
        elseif is_macrocall(stmt, Symbol("@private")) && length(stmt.args) == 3 &&
                isexpr(stmt.args[3], :(=))
            # `@private lhs = rhs`
            take_line!(allocations)
            push!(allocations, stmt.args[3])
            continue
        elseif @capture(stmt, lhs_ = rhs_ | (vs__, lhs_ = rhs_))
            if is_macrocall(rhs, Symbol("@localmem")) || is_macrocall(rhs, Symbol("@uniform"))
                take_line!(allocations)
                push!(allocations, stmt)
                continue
            elseif is_macrocall(rhs, Symbol("@private")) && length(rhs.args) == 4
                # Implement the legacy `mem = @private T dims` as
                # mem = Scratchpad(T, Val(dims))
                T, dims = rhs.args[3], rhs.args[4]
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

    # everything since the last `@synchronize`, including hoisted statements
    if !isempty(current) || !isempty(allocations)
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
