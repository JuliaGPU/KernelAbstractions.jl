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
function __kernel(expr, force_inbounds = false)
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

    def_gpu = deepcopy(def)
    def_gpu[:name] = gpu_name = Symbol(:gpu_, name)
    transform_gpu!(def_gpu, constargs, force_inbounds)
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
function transform_gpu!(def, constargs, force_inbounds)
    let_constargs = Expr[]
    for (i, arg) in enumerate(def[:args])
        if constargs[i]
            push!(let_constargs, :($arg = $constify($arg)))
        end
    end
    pushfirst!(def[:args], :__ctx__)
    body = def[:body]
    if force_inbounds
        body = quote
            @inbounds $(body)
        end
    end
    body = quote
        if $__validindex(__ctx__)
            $(body)
        end
        return nothing
    end
    def[:body] = Expr(
        :let,
        Expr(:block, let_constargs...),
        body,
    )
end