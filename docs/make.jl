using Documenter, KernelAbstractions

makedocs(
    modules = [KernelAbstractions],
    sitename = "KernelAbstractions",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "Home"    => "index.md",
        "Writing kernels" => "kernels.md",
        "Examples"     => [
            "examples/memcopy.md"
        ],
        "API"          => "api.md",
    ],
    doctest = true
)
