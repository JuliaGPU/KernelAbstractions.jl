import Base.Meta: isexpr

# XXX: Proper errors
function __kernel(expr)
    @assert isexpr(expr, :function)
    decl = expr.args[1]
    body = expr.args[2]

    # parse decl
    @assert isexpr(decl, :call)
    name = decl.args[1]

    # List of tuple (Symbol, Bool) where the bool
    # marks if the arg is const
    args = Any[]
    for i in 2:length(decl.args)
        arg = decl.args[i]
        if isexpr(arg, :macrocall)
            if arg.args[1] === Symbol("@Const")
                # args[2] is a LineInfo node
                push!(args, (arg.args[3], true))
                continue
            end
        end
        push!(args, (arg, false))
    end

    arglist = map(a->a[1], args)

    # create two functions
    # 1. GPU function
    # 2. CPU function with work-group loops inserted
    gpu_name = esc(gensym(Symbol(:gpu_, name)))
    cpu_name = esc(gensym(Symbol(:cpu_, name)))
    alloc_name = esc(gensym(Symbol(:alloc_, name)))
    name = esc(name)

    gpu_decl = Expr(:call, gpu_name, arglist...)
    cpu_decl = Expr(:call, cpu_name, arglist...)
    alloc_decl = Expr(:call, alloc_name, arglist...)

    # Without the deepcopy we might accidentially modify expr shared between CPU and GPU
    gpu_body = transform_gpu(deepcopy(body), args)
    gpu_function = Expr(:function, gpu_decl, gpu_body)

    cpu_body = transform_cpu(deepcopy(body), args)
    cpu_function = Expr(:function, cpu_decl, cpu_body)

    dynamic_allocator_body = transform_dynamic(body, args)
    dynamic_allocator_function = Expr(:function, alloc_decl, dynamic_allocator_body)

    # create constructor functions
    constructors = quote
        $name(dev::Device) = $name(dev, $DynamicSize(), $DynamicSize())
        $name(dev::Device, size) = $name(dev, $StaticSize(prod(size)), $DynamicSize())
        $name(dev::Device, size, range) = $name(dev, $StaticSize(prod(size)), $StaticSize(range))
        function $name(::Device, ::S, ::NDRange) where {Device<:$CPU, S<:$_Size, NDRange<:$_Size}
            return $Kernel{Device, S, NDRange, typeof($cpu_name)}($cpu_name)
        end
        function $name(::Device, ::S, ::NDRange) where {Device<:$GPU, S<:$_Size, NDRange<:$_Size}
            return $Kernel{Device, S, NDRange, typeof($gpu_name)}($gpu_name)
        end
        KernelAbstractions.allocator(::typeof($gpu_name)) = $alloc_name
        KernelAbstractions.allocator(::typeof($cpu_name)) = $alloc_name
    end

    return Expr(:toplevel, cpu_function, gpu_function, dynamic_allocator_function, constructors)
end

# Transform function for GPU execution
# This involves marking constant arguments
function transform_gpu(expr, args)
    new_stmts = Expr[]
    for (arg, isconst) in args
        if isconst
            push!(new_stmts, :($arg = $ConstWrapper($arg)))
        end
    end
    return quote
        if $__validindex()
            $(new_stmts...)
            $expr
        end
        return nothing
    end
end

function split(stmts)
    # 1. Split the code into blocks separated by `@synchronize`
    # 2. Aggregate the index and allocation expressions seen at the sync points
    indicies    = Any[]
    allocations = Any[]
    dynamic_allocs = Any[]
    loops       = Any[]
    current     = Any[]

    for stmt in stmts.args
        if isexpr(stmt, :macrocall) && stmt.args[1] === Symbol("@synchronize")
            push!(loops, (current, deepcopy(indicies), allocations))
            allocations = Any[]
            current     = Any[]
            continue
        elseif isexpr(stmt, :(=))
            rhs = stmt.args[2]
            if isexpr(rhs, :macrocall)
                callee = rhs.args[1]
                if callee === Symbol("@index")
                    push!(indicies, stmt)
                    continue
                elseif callee === Symbol("@localmem") || callee === Symbol("@private")
                    push!(allocations, stmt)
                    continue
                elseif callee === Symbol("@dynamic_localmem")
                    # args[2] LineNumberNode
                    Tvar = rhs.args[3]
                    size = rhs.args[4]
                    # add all previous dynamic allocations
                    append!(rhs.args, dynamic_allocs)
                    push!(allocations, stmt)
                    push!(dynamic_allocs, Expr(:tuple, Tvar, size))
                    continue
                end
            end
        end

        if isexpr(stmt, :block)
            # XXX: What about loops, let, ...
            @warn "Encountered a block at the top-level unclear semantics"
        end
        push!(current, stmt)
    end

    # everything since the last `@synchronize`
    if !isempty(current)
        push!(loops, (current, copy(indicies), allocations))
    end
    return loops
end

function generate_cpu_code(loops)
    # Create loops
    new_stmts = Any[]
    for (body, indexes, allocations) in loops
        idx = gensym(:I)
        # splice index into the i = @index(Cartesian, $idx)
        for stmt in indexes
            @assert stmt.head === :(=)
            rhs = stmt.args[2]
            push!(rhs.args, idx)
        end
        loop = quote
            $(allocations...)
            for $idx in $__workitems_iterspace()
                $__validindex($idx) || continue
                $(indexes...)
                $(body...)
            end
        end
        push!(new_stmts, loop)
    end
    return Expr(:block, new_stmts...)
end

function transform_cpu(stmts, args)
    new_stmts = Expr[]
    for (arg, isconst) in args
        if isconst
            # XXX: Deal with OffsetArrays
            push!(new_stmts, :($arg = $Base.Experimental.Const($arg)))
        end
    end
    loops = split(stmts)
    body  = generate_cpu_code(loops) 

    # push!(new_stmts, Expr(:aliasscope))
    push!(new_stmts, body)
    # push!(new_stmts, Expr(:popaliasscope))
    push!(new_stmts, :(return nothing))
    return Expr(:block, new_stmts...)
end

function transform_dynamic(stmts, args)
    # 1. Find dynamic allocations
    allocators = Expr[]
    for stmt in stmts.args
        if isexpr(stmt, :(=))
            rhs = stmt.args[2]
            if isexpr(rhs, :macrocall)
                callee = rhs.args[1]
                if callee === Symbol("@dynamic_localmem")
                    # args[2] LineNumberNode
                    Tvar = rhs.args[3]
                    size = rhs.args[4]
                    push!(allocators, Expr(:tuple, Expr(:call, :sizeof, Tvar), size))
                    continue
                end
            end
        end
    end
    return Expr(:block, Expr(:tuple, allocators...))
end


