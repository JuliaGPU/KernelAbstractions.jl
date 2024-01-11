# Memcopy with static NDRange

The first example simple copies memory from `B` to `A`. In contrast to the previous examples
it uses a fully static kernel configuration. Specializing the kernel on the iteration range itself.

````@eval
using Markdown
using KernelAbstractions
path = joinpath(dirname(pathof(KernelAbstractions)), "..", "examples/memcopy_static.jl")
Markdown.parse("""
```julia
$(read(path, String))
```
""")
````
