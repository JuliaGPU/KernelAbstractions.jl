module Extras

include("loopinfo.jl")
using .LoopInfo
export @unroll

include("LIKWID.jl")

end # module
