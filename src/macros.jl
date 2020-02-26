import MacroTools: splitdef, combinedef, isexpr

# XXX: Proper errors
function __kernel(expr)
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
        if !@isdefined($name)
            $name(dev::$Device) = $name(dev, $DynamicSize(), $DynamicSize())
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
    new_stmts = Expr[]
    for (i, arg) in enumerate(def[:args])
        if constargs[i]
            push!(new_stmts, :($arg = $constify($arg)))
        end
    end

    def[:body] = quote
        if $__validindex()
            $(new_stmts...)
            $(def[:body])
        end
        return nothing
    end
end

# The hard case, transform the function for CPU execution
# - mark constant arguments by applying `constify`.
# - insert aliasscope markers
# - insert implied loop bodys
#   - handle indicies
#   - hoist workgroup definitions
#   - hoist uniform variables
function transform_cpu!(def, constargs)
    new_stmts = Expr[]
    for (i, arg) in enumerate(def[:args])
        if constargs[i]
            push!(new_stmts, :($arg = $constify($arg)))
        end
    end

    body = MacroTools.flatten(def[:body])
    loops = split(body)

    push!(new_stmts, Expr(:aliasscope))
    for loop in loops
        push!(new_stmts, emit(loop))
    end
    push!(new_stmts, Expr(:popaliasscope))
    push!(new_stmts, :(return nothing))
    def[:body] = Expr(:block, new_stmts...)
end

struct WorkgroupLoop
    indicies :: Vector{Any}
    stmts :: Vector{Any}
    allocations :: Vector{Any}
end


function split(stmts)
    # 1. Split the code into blocks separated by `@synchronize`
    # 2. Aggregate `@index` expressions
    # 3. Hoist allocations
    # 4. Hoist uniforms

    current     = Any[]
    indicies    = Any[]
    allocations = Any[]

    loops = WorkgroupLoop[]
    for stmt in stmts.args
        if isexpr(stmt, :macrocall) && stmt.args[1] === Symbol("@synchronize")
            loop = WorkgroupLoop(deepcopy(indicies), current, allocations)
            push!(loops, loop)
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
                elseif callee === Symbol("@localmem") ||
                       callee === Symbol("@private")  ||
                       callee === Symbol("@uniform")
                    push!(allocations, stmt)
                    continue
                end
            end
        end

        push!(current, stmt)
    end

    # everything since the last `@synchronize`
    if !isempty(current)
        push!(loops, WorkgroupLoop(deepcopy(indicies), current, allocations))
    end
    return loops
end

function emit(loop)
    idx = gensym(:I)
    for stmt in loop.indicies
        # splice index into the i = @index(Cartesian, $idx)
        @assert stmt.head === :(=)
        rhs = stmt.args[2]
        push!(rhs.args, idx)
    end
    quote
        $(loop.allocations...)
        for $idx in $__workitems_iterspace()
            $__validindex($idx) || continue
            $(loop.indicies...)
            $(loop.stmts...)
        end
    end
end
