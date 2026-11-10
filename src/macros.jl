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
                isexpr(args[i], :...) && error("`@Const` cannot be applied to varargs")
                constargs[i] = true
                continue
            end
        end
        constargs[i] = false
    end

    def_gpu = deepcopy(def)
    def_gpu[:name] = gpu_name = Symbol(:gpu_, name)
    # a kernel returns `nothing` anyway, but a `ConstArgs` query returns its mask
    delete!(def_gpu, :rtype)
    forwarders = default_forwarders!(def_gpu, gpu_name, constargs)
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
    # Julia takes a method's location from the first `LineNumberNode` of its body, and only
    # tracks coverage of methods whose file is tracked. The transformed body starts with
    # generated statements, which would leave the kernel at `none:0` and thus untracked under
    # `--code-coverage=@path`.
    def_gpu[:body] = Expr(:block, __source__, def_gpu[:body])
    gpu_function = combinedef(def_gpu)

    # Properties of the kernel are methods of its function, rather than of a function of
    # KernelAbstractions, so that a kernel can also be defined in a local scope. Like the
    # constructors, they are defined with the first method of the kernel, since a method can't
    # be overwritten during precompilation.
    queries = Any[]
    if find_collective(def[:body])
        # see `uses_collectives`
        push!(queries, :($gpu_name(::$CollectivesQuery) = true))
    end

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
            $(queries...)
        end
    end
    constructors = relocate_lines(constructors, __source__)

    return Expr(:block, esc(gpu_function), esc(forwarders), esc(constructors))
end

"""
    GeneratedKernelError{Msg}

Marker type a `generated=true` kernel returns when its generator failed, carrying the
error message in its type parameter. A kernel that returns this shows up in the
`KernelError` GPUCompiler raises for kernels that return a value, which is the only
channel through which a failure at generation time can be reported. The type prints
as `GeneratedKernelError{Symbol("message")}`.
"""
struct GeneratedKernelError{Msg} end
# Overloading `show` on the type for nicer output would invalidate thousands of
# `show(::IO, ::Type)` callers when the package loads.
GeneratedKernelError(msg::AbstractString) = GeneratedKernelError{Symbol(msg)}()

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

# the answer of a kernel method to a `ConstArgs` query
constargs_mask(constargs) = any(constargs) ? :(Val($(Tuple(constargs)))) : :nothing

whereparam_name(p) = isexpr(p, :<:) || isexpr(p, :>:) ? p.args[1] : isexpr(p, :comparison) ? p.args[3] : p

uses(ex, name) = ex === name || (ex isa Expr && any(arg -> uses(arg, name), ex.args))

# The methods for the arities that default arguments allow, which Julia would otherwise
# generate itself. Like those, each supplies the next default, so that a default can refer to
# the ones before it, but these first answer `ConstArgs` queries, so that a query doesn't
# evaluate the defaults on the host. Strips the defaults from `def`.
function default_forwarders!(def, gpu_name, constargs)
    # `def` holds values of any type; assert the vector, so that the reductions below don't
    # depend on generic iteration, which packages invalidate by extending it
    args = def[:args]::Vector
    first_default = findfirst(arg -> isexpr(arg, :kw), args)
    first_default === nothing && return Expr(:block)
    nfixed = count(arg -> !isexpr(arg, :...), args)
    # the forwarders pass on their arguments, so they need names
    named = map(args) do arg
        arg = isexpr(arg, :kw) ? arg.args[1] : arg
        return isexpr(arg, :(::)) && length(arg.args) == 1 ? :($(gensym())::$(arg.args[1])) : arg
    end
    argname(arg) = isexpr(arg, :(::)) ? arg.args[1] : arg
    whereparams = collect(Base.get(def, :whereparams, ()))
    forwarders = map((first_default - 1):(nfixed - 1)) do n
        kept = named[1:n]
        # Julia warns about static parameters that a method doesn't use
        used = filter(p -> any(arg -> uses(arg, whereparam_name(p)), kept), whereparams)
        while true
            bounds = filter(p -> !(p in used) && any(q -> uses(q, whereparam_name(p)), used), whereparams)
            isempty(bounds) && break
            append!(used, bounds)
        end
        default = args[n + 1].args[2]
        fdef = Dict{Symbol, Any}(
            :name => gpu_name,
            :args => Any[:(__ctx__::Union{$CompilerMetadata, $ConstArgs}), kept...],
            :kwargs => Any[],
            :whereparams => filter(in(used), whereparams),
            :body => quote
                __ctx__ isa $ConstArgs && return $(constargs_mask(constargs[1:n]))
                # as in the methods Julia generates: referring to the function by name
                # would capture it when it's a closure
                return var"#self#"(__ctx__, $(map(argname, kept)...), $default)
            end,
        )
        return combinedef(fdef)
    end
    def[:args] = map(arg -> isexpr(arg, :kw) ? arg.args[1] : arg, args)
    return Expr(:block, forwarders...)
end

# The easy case, transform the function for GPU execution
# - answer which arguments are marked `@Const` (see `ConstArgs`)
function transform_gpu!(def, constargs, force_inbounds, unsafe_indices)
    # typed, so that the queries of `@kernel` (e.g. `uses_collectives`) never match the kernel
    def[:args] = Any[:(__ctx__::Union{$CompilerMetadata, $ConstArgs}), def[:args]...]
    # `Any[]`, since `split` hands back `LineNumberNode`s alongside `Expr`s
    new_stmts = Any[]
    push!(new_stmts, :(__ctx__ isa $ConstArgs && return $(constargs_mask(constargs))))
    body = MacroTools.flatten(def[:body])
    if !unsafe_indices
        push!(new_stmts, :(__active_lane__ = $__validindex(__ctx__)))
    end
    if force_inbounds
        push!(new_stmts, Expr(:inbounds, true))
    end
    if !unsafe_indices
        lowered = split(body.args)
        check_masked_conditions(body, lowered)
        append!(new_stmts, lowered)
    else
        push!(new_stmts, body)
    end
    if force_inbounds
        push!(new_stmts, Expr(:inbounds, :pop))
    end
    push!(new_stmts, :(return nothing))
    def[:body] = Expr(:block, new_stmts...)
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
function split_branches(expr::Expr, recurse)
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

function find_collective(stmt)
    result = Ref(false)
    postwalk(stmt) do expr
        result[] |= is_collective(expr)
        expr
    end
    return result[]
end

# Whether `stmt` contains a `@synchronize`, or a collective like `@groupreduce` that all
# work-items of the workgroup have to reach as well.
function find_sync(stmt)
    result = Ref(false)
    postwalk(stmt) do expr
        result[] |= is_sync(expr) || is_collective(expr)
        expr
    end
    return result[]
end

# Flush the pending `LineNumberNode` into `stmts`, returning `nothing` to clear it.
function take_line!(stmts, line)
    line === nothing || push!(stmts, line)
    return nothing
end

# Recurse into scope constructs
# TODO: This currently implements hard scoping
#       probably need to implement soft scoping
#       by not deepcopying the environment.
split_scopes(x) = x
function split_scopes(expr::Expr)
    expr = unblock_lines(expr)
    is_collective(expr) && collective_error(expr)
    if expr.head in (:if, :elseif) && find_sync(expr)
        return split_branches(expr, split_scopes)
    elseif is_scope_construct(expr) && any(find_sync, expr.args)
        return Expr(expr.head, split(expr.args)...)
    else
        return Expr(expr.head, map(split_scopes, expr.args)...)
    end
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

    for stmt in stmts
        if stmt isa LineNumberNode
            line = stmt
            continue
        end

        if is_collective_stmt(stmt)
            # executed by all work-items, the padding ones contribute the neutral element
            loop = WorkgroupLoop(current, allocations, false, nothing)
            push!(new_stmts, emit(loop))
            allocations = Any[]
            current = Any[]
            line = take_line!(new_stmts, line)
            push!(new_stmts, mask_collective(stmt))
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

            line = take_line!(new_stmts, line)
            push!(new_stmts, split_scopes(stmt))
            continue
        end

        if is_macrocall(stmt, Symbol("@uniform"))
            line = take_line!(allocations, line)
            push!(allocations, stmt)
            continue
        elseif is_macrocall(stmt, Symbol("@private")) && length(stmt.args) == 3 &&
                isexpr(stmt.args[3], :(=))
            # `@private lhs = rhs`
            line = take_line!(allocations, line)
            push!(allocations, stmt.args[3])
            continue
        elseif @capture(stmt, lhs_ = rhs_ | (vs__, lhs_ = rhs_))
            if is_macrocall(rhs, Symbol("@localmem")) || is_macrocall(rhs, Symbol("@uniform"))
                line = take_line!(allocations, line)
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
                line = take_line!(allocations, line)
                push!(allocations, :($lhs = $alloc))
                continue
            end
        end

        line = take_line!(current, line)
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

# Control flow that contains a `@synchronize` runs on all work-items, including the padding
# ones, which skip the code that `emit` guards with `__active_lane__`. A condition that reads
# a variable assigned in guarded code therefore reads it undefined (or stale) on the padding
# work-items, which then diverge from the others at the `@synchronize`. Reject such kernels.
function check_masked_conditions(body, lowered)
    conditions = Tuple{Expr, Any, Union{Nothing, LineNumberNode}}[]
    collect_conditions!(conditions, body, nothing)
    isempty(conditions) && return
    masked = Set{Symbol}()
    foreach(stmt -> collect_masked!(masked, stmt), lowered)
    isempty(masked) && return
    for (construct, cond, line) in conditions
        reads = Set{Symbol}()
        collect_reads!(reads, cond)
        vars = intersect(reads, masked)
        isempty(vars) && continue
        var = minimum(vars)
        kind = construct.head === :elseif ? :if : construct.head
        location = line === nothing ? "" : " at $(line.file):$(line.line)"
        error(
            "`@kernel`: the `$kind`$location contains a `@synchronize`, so its condition " *
                "runs on all work-items, including those outside the ndrange. It reads `$var`, " *
                "which is assigned in code those work-items skip: compute `$var` with " *
                "`@uniform` wherever it is assigned or updated."
        )
    end
    return
end

# The conditions of the `while`, `for` and `if` constructs in `expr` that contain a
# `@synchronize` or a collective, as `(construct, condition, line)`.
function collect_conditions!(conditions, expr, line)
    expr isa Expr || return
    if expr.head === :block
        for arg in expr.args
            if arg isa LineNumberNode
                line = arg
            else
                collect_conditions!(conditions, arg, line)
            end
        end
        return
    elseif expr.head === :while && find_sync(expr.args[2])
        push!(conditions, (expr, expr.args[1], line))
    elseif expr.head === :for && find_sync(expr.args[2])
        specs = isexpr(expr.args[1], :block) ? expr.args[1].args : Any[expr.args[1]]
        for spec in specs
            spec isa LineNumberNode && continue
            push!(conditions, (expr, spec.args[2], line))
        end
    elseif expr.head in (:if, :elseif) && any(find_sync, expr.args[2:end])
        push!(conditions, (expr, expr.args[1], line))
    end
    for arg in expr.args
        collect_conditions!(conditions, arg, line)
    end
    return
end

# The variables assigned in the code of `lowered` that `emit` guards with `__active_lane__`.
function collect_masked!(masked, expr)
    expr isa Expr || return
    if expr.head === :if && expr.args[1] === :__active_lane__
        collect_assigned!(masked, expr.args[2])
    else
        foreach(arg -> collect_masked!(masked, arg), expr.args)
    end
    return
end

# Heads whose `=` arguments are not assignments: keyword arguments and named tuple fields.
# (Those of macro calls are, e.g. `@inbounds x = A[i]`.)
const NON_ASSIGNMENT_HEADS = (:call, :tuple, :parameters, :ref, :curly)
# Heads that introduce a scope of their own, whose assignments are local to it.
const SCOPE_HEADS = (:->, :function, :let, :generator, :comprehension, :typed_comprehension)

function collect_assigned!(assigned, expr)
    expr isa Expr || return
    expr.head in SCOPE_HEADS && return
    if expr.head === :for
        # the iteration variables are local to the loop
        collect_assigned!(assigned, expr.args[2])
        return
    end
    if expr.head === :(=) || (expr.head isa Symbol && is_update_head(expr.head))
        collect_lhs!(assigned, expr.args[1])
        collect_assigned!(assigned, expr.args[2])
        return
    end
    for arg in expr.args
        if expr.head in NON_ASSIGNMENT_HEADS && isexpr(arg, :(=))
            collect_assigned!(assigned, arg.args[2])
        else
            collect_assigned!(assigned, arg)
        end
    end
    return
end

# `+=`, `|=`, etc. (but not comparisons like `<=`, nor `.=` and `.+=`, which mutate)
function is_update_head(head::Symbol)
    s = String(head)
    startswith(s, ".") && return false
    return endswith(s, "=") && !(head in (:(==), :(!=), :(<=), :(>=), :(===), :(!==))) &&
        length(s) > 1 && Base.isoperator(Symbol(chop(s)))
end

collect_lhs!(assigned, lhs::Symbol) = push!(assigned, lhs)
function collect_lhs!(assigned, lhs::Expr)
    if lhs.head in (:tuple, :parameters)
        foreach(arg -> collect_lhs!(assigned, arg), lhs.args)
    elseif lhs.head === :(::)
        collect_lhs!(assigned, lhs.args[1])
    end
    # `A[i] = ...` and `x.f = ...` mutate rather than assign
    return
end
collect_lhs!(assigned, lhs) = nothing

# The variables that `expr` reads.
collect_reads!(reads, expr::Symbol) = push!(reads, expr)
function collect_reads!(reads, expr::Expr)
    expr.head in SCOPE_HEADS && return
    if expr.head === :.
        # `x.f` reads `x`, not `f`
        collect_reads!(reads, expr.args[1])
        return
    elseif expr.head === :macrocall
        foreach(arg -> collect_reads!(reads, arg), expr.args[2:end])
        return
    end
    foreach(arg -> collect_reads!(reads, arg), expr.args)
    return
end
collect_reads!(reads, expr) = nothing
