using KernelAbstractions
using Documenter

makedocs(;
    modules=[KernelAbstractions],
    authors="JuliaGPU and contributors",
    repo="https://github.com/JuliaGPU/KernelAbstractions.jl/blob/{commit}{path}#L{line}",
    sitename="KernelAbstractions.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://juliagpu.github.io/KernelAbstractions.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Writing kernels" => "kernels.md",
        "Examples" => [
            "examples/memcopy.md",
            "examples/memcopy_static.md",
            "examples/naive_transpose.md",
            "examples/performance.md",
            "examples/matmul.md",
        ], # Examples
        "API" => "api.md",
        "Extras" => [
            "extras/unrolling.md",
        ], # Extras
    ], # pages
)

deploydocs(;
    repo="github.com/JuliaGPU/KernelAbstractions.jl",
)
