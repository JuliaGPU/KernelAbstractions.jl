# Matmul


````@eval
using Markdown
using KernelAbstractions
path = joinpath(dirname(pathof(KernelAbstractions)), "..", "examples/matmul.jl")
Markdown.parse("""
```julia
$(read(path, String))
```
""")
````
