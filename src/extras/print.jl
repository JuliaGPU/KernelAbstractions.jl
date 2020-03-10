@generated function cuda_print(items...)
    args = Union{Val,Expr,Symbol}[]

    parts = [parts...]
    while true
        isempty(parts) && break

        part = popfirst!(parts)

        # handle string interpolation
        if isa(part, Expr) && part.head == :string
            parts = vcat(part.args, parts)
            continue
        end

        # expose literals to the generator by using Val types
        if isbits(part) # literal numbers, etc
            push!(args, Val(part))
        elseif isa(part, QuoteNode) # literal symbols
            push!(args, Val(part.value))
        elseif isa(part, String) # literal strings need to be interned
            push!(args, Val(Symbol(part)))
        else # actual values that will be passed to printf
            push!(args, part)
        end
    end

    quote
        CUDAnative._cuprint($(map(esc, args)...))
    end
end
