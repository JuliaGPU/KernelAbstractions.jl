# Writing kernels

These kernel language constructs are intended to be used inside [`@kernel`](@ref) functions.
They are not valid in ordinary Julia code (except when using experimental `@kernel cpu=false`).

## Constant arguments

Kernel functions allow input arguments to be marked with the [`@Const`](@ref) macro. It informs
the compiler that the memory accessed through that argument will not be written to as part of
the kernel, and that it does not alias any other memory in the kernel. If you are used to CUDA C,
this is similar to `const restrict`.

```julia
@kernel function saxpy!(a, @Const(X), Y)
    I = @index(Global)
    @inbounds Y[I] = a * X[I] + Y[I]
end
```

## Indexing

The [`@index`](@ref) macro returns the index of the current work item. Choose a **granularity**
and an optional **kind**:

| Granularity | Meaning |
|-------------|---------|
| `Global` | Index over the full `ndrange` (use for global memory) |
| `Group` | Index of the current workgroup |
| `Local` | Index within the current workgroup |

| Kind | Result type |
|------|-------------|
| `Linear` (default) | `Int` linear index |
| `Cartesian` | `CartesianIndex` for multi-dimensional `ndrange` |
| `NTuple` | `NTuple` of `Int` indices |

```julia
@kernel function fill_diagonal!(A, val)
    I = @index(Global, Cartesian)
    if I[1] == I[2]
        @inbounds A[I] = val
    end
end

@kernel function linear_example(A)
    I = @index(Global, Linear)   # 1, 2, 3, ...
    g = @index(Group, Linear)    # workgroup id
    l = @index(Local, Linear)    # lane within workgroup
    @inbounds A[I] = g + l
end
```

Inside a kernel, [`@groupsize`](@ref) and [`@ndrange`](@ref) query the launch configuration:

```julia
@kernel function scale!(A, factor)
    N = prod(@groupsize())
    I = @index(Global, Linear)
    lmem = @localmem Float32 (N,)
    i = @index(Local, Linear)
    lmem[i] = factor
    @synchronize()
    @inbounds A[I] = lmem[i]
end
```

## Local memory, synchronization, and private memory

[`@localmem`](@ref) declares storage shared by all work items in a workgroup. Reads and writes
must be separated by [`@synchronize`](@ref) if they are performed by different work items:

```julia
@kernel function reverse_block!(A)
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    N = prod(@groupsize())
    buf = @localmem Int (N,)
    buf[i] = i
    @synchronize()
    @inbounds A[I] = buf[N - i + 1]
end
```

[`@private`](@ref) and [`@uniform`](@ref) are deprecated for KernelAbstractions 1.0. Prefer
`MArray` for per-lane scratch storage that does not need to survive across `@synchronize`.

## Launching kernels

Construct a kernel by calling the kernel function on a backend and optional static sizes, then
launch it with `ndrange`:

```julia
# dynamic sizes — supply ndrange (and optionally workgroupsize) at launch
kernel = my_kernel(backend)
kernel(A, ndrange=size(A))

# static workgroup size
kernel = my_kernel(backend, 256)
kernel(A, ndrange=size(A))

# static workgroup size and ndrange — fewer runtime checks, may reduce recompilation
kernel = my_kernel(backend, 32, size(A))
kernel(A)
```

On GPU backends, obtain the backend from an array with [`get_backend`](@ref) and always call
[`synchronize`](@ref) before reading results on the host. See the [Quickstart](@ref) for a full walkthrough and the Examples section of the manual
for larger patterns.
