module KernelIntrinsics

"""
    get_global_size()::@NamedTuple{x::Int32, y::Int32, z::Int32}

Return the number of global work-items specified.
"""
function get_global_size end

"""
    get_global_id()::@NamedTuple{x::Int32, y::Int32, z::Int32}

Returns the unique global work-item ID.

!!! note
    1-based.
"""
function get_global_id end

"""
    get_local_size()::@NamedTuple{x::Int32, y::Int32, z::Int32}

Return the number of local work-items specified.
"""
function get_local_size end

"""
    get_local_id()::@NamedTuple{x::Int32, y::Int32, z::Int32}

Returns the unique local work-item ID.

!!! note
    1-based.
"""
function get_local_id end

"""
    get_num_groups()::@NamedTuple{x::Int32, y::Int32, z::Int32}

Returns the number of groups.
"""
function get_num_groups end

"""
    get_group_id()::@NamedTuple{x::Int32, y::Int32, z::Int32}

Returns the unique group ID.

!!! note
    1-based.
"""
function get_group_id end

function localmemory end
function barrier end
function print end

end
