# Kernel config struct

export Config

struct Config{
    GROUPSIZE, 
    MAX_NDRANGE,

    ITEMS_PER_WORKITEM,
    USE_ATOMICS,
    USE_WARPS
    }

    function Config(groupsize, max_ndrange, items_per_workitem , use_atomics, use_warps)
        new{groupsize, max_ndrange, items_per_workitem, use_atomics, use_warps}()
    end
end

@inline function Base.getproperty(conf::Config{GROUPSIZE, MAX_NDRANGE, ITEMS_PER_WORKITEM, USE_ATOMICS, USE_WARPS}, sym::Symbol) where { GROUPSIZE, MAX_NDRANGE,ITEMS_PER_WORKITEM, USE_ATOMICS, USE_WARPS }

    if sym == :groupsize
        GROUPSIZE
    elseif sym == :max_ndrange
        MAX_NDRANGE
    elseif sym == :items_per_workitem
        ITEMS_PER_WORKITEM
    elseif sym == :use_atomics
        USE_ATOMICS
    elseif sym == :use_warps
        USE_WARPS        
    else
        # fallback for nothing
        getfield(conf, sym)
    end
end