module Extras

include("loopinfo.jl")
using .LoopInfo
export @unroll

include("timeline.jl")
using .Timeline
export Timeline

end # module
