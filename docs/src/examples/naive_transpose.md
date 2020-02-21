# Naive Transpose

````@eval
using Markdown
using KernelAbstractions
path = joinpath(dirname(pathof(KernelAbstractions)), "..", "examples/naive_transpose.jl")
Markdown.parse("""
```julia
$(read(path, String))
```
""")
````
