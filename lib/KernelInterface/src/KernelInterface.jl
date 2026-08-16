"""
# `KernelInterface`

The `KernelInterface` (or `KI`) module defines the API interface for backends to define various lower-level device and
host-side functionality. The `KI` interface is used to define the higher-level device-side
functionality in `KernelAbstractions`.

Both provide APIs for host and device-side functionality, but `KI` focuses on on lower-level
functionality that is shared amongst backends, while `KernelAbstractions` provides higher-level functionality
such as writing kernels that work on arrays with an arbitrary number of dimensions, or convenience functions
like allocating arrays on a backend.
"""
module KernelInterface

include("utils.jl")

include("backend.jl")
include("device.jl")
include("host.jl")

end
