using Documenter, KernelAbstractions

makedocs(
    modules = [KernelAbstractions],
    format = :html,
    sitename = "KernelAbstractions.jl",
    pages = [
        "Home"    => "index.md",
        "Kernel Language" => "kernels.md",
        "Design" => "design.md"
    ],
    doctest = true
)
