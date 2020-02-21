using Documenter, KernelAbstractions

makedocs(
    modules = [KernelAbstractions],
    sitename = "KernelAbstractions",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "Home" => "index.md",
        "Writing kernels" => "kernels.md",
        "Examples" => [
            "examples/memcopy.md"
            "examples/memcopy_static.md"
            "examples/naive_transpose.md"
            "examples/performance.md"
            "examples/matmul.md"
        ],
        "API"          => "api.md",
        "Extras" => [
            "extras/unrolling.md"
        ]
    ],
    doctest = true
)
