module LoopInfo

using MacroTools
export @unroll

##
# Uses the loopinfo expr node to attach LLVM loopinfo to loops
# the full list of supported metadata nodes is available at
# https://llvm.org/docs/LangRef.html#llvm-loop
# TODO: Figure out how to deal with compile-time constants in `@unroll(N, expr)`
#       so constants that come from `Val{N}` but are not parse time constant.
#       Most likely will require changes to base Julia.
##

module MD
    unroll_count(n) = (Symbol("llvm.loop.unroll.count"), convert(Int, n))
    unroll_disable() = (Symbol("llvm.loop.unroll.disable"), 1)
    unroll_enable() = (Symbol("llvm.loop.unroll.enable"), 1)
    unroll_full() = (Symbol("llvm.loop.unroll.full"), 1)
end

function loopinfo(expr, nodes...)
    if @capture(expr, for i_ in iter_ body__ end)
        return quote
            for $i in $iter
                $(body...)
                $(Expr(:loopinfo, nodes...))
            end
        end
    else
        error("Syntax error: loopinfo needs a for loop")
    end
end

"""
    @unroll expr

Takes a for loop as `expr` and informs the LLVM unroller to fully unroll it, if
it is safe to do so and the loop count is known.
"""
macro unroll(expr)
    expr = loopinfo(expr, MD.unroll_full())
    return esc(expr)
end

"""
    @unroll N expr

Takes a for loop as `expr` and informs the LLVM unroller to unroll it `N` times,
if it is safe to do so.
"""
macro unroll(N, expr)
    if !(N isa Integer)
        @debug "@unroll macro inputs" N expr
        error("Syntax error: `@unroll N expr` needs a constant integer N")
    end
    expr = loopinfo(expr, MD.unroll_count(N))
    return esc(expr)
end

end #module
