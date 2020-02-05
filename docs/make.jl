using Documenter, KernelAbstractions

makedocs(
    modules = [KernelAbstractions],
    sitename = "KernelAbstractions.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "Home"    => "index.md",
        "Kernel Language" => "kernels.md",
        "Design" => "design.md"
    ],
    doctest = true
)
