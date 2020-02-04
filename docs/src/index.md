# KernelAbstractions.jl

```@contents
```

## Kernel functions 

```@docs
@kernel
```

### Marking input arguments as constant

```@docs
@Const
```

## Important difference to Julia

- Functions inside kernels are forcefully inlined, except when marked with `@noinline`
- Floating-point multiplication, addition, subtraction are marked contractable

## Examples
### Memory copy

The first example simple copies memory from `A` to `B`


````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../examples/memcopy.jl", String))
```
""")
````

## How to debug kernels

## How to profile kernels
