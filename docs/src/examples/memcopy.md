# Memcopy

The first example simple copies memory from `B` to `A`

````@eval
using Markdown
using KernelAbstractions
path = joinpath(dirname(pathof(KernelAbstractions)), "..", "examples/memcopy.jl")
Markdown.parse("""
```julia
$(read(path, String))
```
""")
````
