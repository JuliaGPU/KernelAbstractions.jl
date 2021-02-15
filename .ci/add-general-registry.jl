import Pkg

if VERSION >= v"1.5-" && !isdir(joinpath(DEPOT_PATH[1], "registries", "General"))
    Pkg.Registry.add("General")
end
