# Develop a back end from a branch, for testing it against this checkout of KernelAbstractions
# and KernelInterface:
#
#     julia develop_backend.jl URL BRANCH [SUBDIR...]
#
# The back end is cloned (with the packages in SUBDIR), and `[sources]` entries of its projects
# for KernelAbstractions or KernelInterface are redirected to this checkout. Julia 1.12+
# honours those `[sources]`, also in the back end's test environment, which would otherwise
# test whatever version they point to.

using Pkg, TOML

const checkout = dirname(@__DIR__)
const packages = Dict(
    "KernelAbstractions" => checkout,
    "KernelInterface" => joinpath(checkout, "lib", "KernelInterface"),
)

url, branch, subdirs... = ARGS
dir = joinpath(checkout, ".backend")
rm(dir; force = true, recursive = true)
run(`git clone --quiet --depth 1 --branch $branch $url $dir`)

for (root, _, files) in walkdir(dir), file in files
    file in ("Project.toml", "JuliaProject.toml") || continue
    path = joinpath(root, file)
    project = TOML.parsefile(path)
    sources = get(project, "sources", Dict())
    names = intersect(keys(sources), keys(packages))
    isempty(names) && continue
    for name in names
        sources[name] = Dict("path" => packages[name])
    end
    open(io -> TOML.print(io, project), path, "w")
    println("Pointed $(join(names, " and ")) in $(relpath(path, dir)) at this checkout")
end

Pkg.develop([PackageSpec(; path = dir); [PackageSpec(; path = joinpath(dir, s)) for s in subdirs]])
# developing the back end can replace the developed KernelAbstractions and KernelInterface
Pkg.develop([PackageSpec(; name, path) for (name, path) in packages])
