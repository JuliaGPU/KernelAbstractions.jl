module Extras

include("loopinfo.jl")
using .LoopInfo
export @unroll

include("kashim.jl")

end # module
