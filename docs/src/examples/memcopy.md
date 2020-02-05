The first example simple copies memory from `A` to `B`


````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../examples/memcopy.jl", String))
```
""")
````