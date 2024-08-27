push!(Base.LOAD_PATH, dirname(@__DIR__))

using KernelAbstractions
using Documenter

function main()
    ci = get(ENV, "CI", "") == "true"

    makedocs(;
        modules = [KernelAbstractions],
        authors = "JuliaGPU and contributors",
        repo = "https://github.com/JuliaGPU/KernelAbstractions.jl/blob/{commit}{path}#L{line}",
        sitename = "KernelAbstractions.jl",
        format = Documenter.HTML(;
            prettyurls = ci,
            canonical = "https://juliagpu.github.io/KernelAbstractions.jl",
            assets = String[],
        ),
        warnonly = [:missing_docs],
        pages = [
            "Home" => "index.md",
            "Quickstart" => "quickstart.md",
            "Writing kernels" => "kernels.md",
            "Examples" => [
                "examples/memcopy.md",
                "examples/memcopy_static.md",
                "examples/naive_transpose.md",
                "examples/performance.md",
                "examples/matmul.md",
                "examples/numa_aware.md",
                "examples/atomix.md",
            ], # Examples
            "API" => "api.md",
            "Extras" => [
                "extras/unrolling.md",
            ], # Extras
            "Notes for implementations" => "implementations.md",
        ], # pages
    )

    if ci
        deploydocs(;
            repo = "github.com/JuliaGPU/KernelAbstractions.jl",
            push_preview = true,
        )
    end
    return
end

isinteractive() || main()
