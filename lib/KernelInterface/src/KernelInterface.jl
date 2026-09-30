"""
# `KernelInterface`

The `KernelInterface` (or `KI`) module defines the API interface for backends to define various lower-level device and
host-side functionality. The `KI` interface is used to define the higher-level device-side
functionality in `KernelAbstractions`.

Both provide APIs for host and device-side functionality, but `KI` focuses on lower-level
functionality that is shared amongst backends, while `KernelAbstractions` provides higher-level functionality
such as writing kernels that work on arrays with an arbitrary number of dimensions, or convenience functions
like allocating arrays on a backend.
"""
module KernelInterface

include("utils.jl")

include("backend.jl")
include("device.jl")
include("launch.jl")
include("host.jl")

# the public API; nothing is exported, so that `KI.` prefixes the interface everywhere
@static if VERSION >= v"1.11"
    eval(
        Expr(
            :public,
            # backends
            :Backend, :get_backend,
            # device side
            :get_global_id, :get_global_size, :get_local_id, :get_local_size,
            :get_group_id, :get_num_groups,
            :get_sub_group_size, :get_max_sub_group_size, :get_num_sub_groups,
            :get_sub_group_id, :get_sub_group_local_id,
            :localmemory, :shfl_down, :barrier, :sub_group_barrier, :_print,
            # compilation and launch
            :Kernel, :kernel_function, :argconvert, :launch, Symbol("@launch"),
            :launch_configuration, :max_work_group_size, :max_work_group_dims,
            :max_num_groups, :sub_group_size, :multiprocessor_count,
            # host side
            :allocate, :zeros, :ones, :copyto!, :pagelock!, :unsafe_free!,
            :synchronize, :record_event, :wait_event, :priority!,
            :device, :ndevices, :device!,
            :functional, :versioninfo,
            :supports_unified, :supports_atomics, :supports_float64,
            :supports_subgroups, :supports_shuffle,
        )
    )
end

end
