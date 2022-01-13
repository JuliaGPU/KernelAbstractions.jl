import Pkg

if !isdir(joinpath(DEPOT_PATH[1], "registries", "General"))
    Pkg.Registry.add("General")
end
