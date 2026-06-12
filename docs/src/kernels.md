# Writing kernels

These kernel language constructs are intended to be used inside [`@kernel`](@ref) functions.
They are not valid in ordinary Julia code.

## Constant arguments

Kernel functions allow input arguments to be marked with the [`@Const`](@ref) macro. It informs
the compiler that the memory accessed through that argument will not be written to as part of
the kernel, and that it does not alias any other memory in the kernel. If you are used to CUDA C,
this is similar to `const restrict`.

```julia
using KernelAbstractions

@kernel function saxpy!(a, @Const(X), Y)
    I = @index(Global)
    @inbounds Y[I] = a * X[I] + Y[I]
end

a = 2.0
X = collect(1.0:8.0)
Y = fill(1.0, 8)
saxpy!(CPU(), 8, size(Y))(a, X, Y)
Y

# output

8-element Vector{Float64}:
  3.0
  5.0
  7.0
  9.0
 11.0
 13.0
 15.0
 17.0
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

```jldoctest
using KernelAbstractions

@kernel function fill_diagonal!(A, val)
    I = @index(Global, Cartesian)
    if I[1] == I[2]
        @inbounds A[I] = val
    end
end

A = collect(reshape(1.0:16.0, 4, 4))
fill_diagonal!(CPU(), 4, size(A))(A, 42)
A

# output

4×4 Matrix{Float64}:
 42.0   5.0   9.0  13.0
  2.0  42.0  10.0  14.0
  3.0   7.0  42.0  15.0
  4.0   8.0  12.0  42.0
```

```jldoctest
using KernelAbstractions

@kernel function linear_example!(A)
    I = @index(Global, Linear)   # 1, 2, 3, ...
    g = @index(Group, Linear)    # workgroup id
    l = @index(Local, Linear)    # lane within workgroup
    @inbounds A[I] = g + l
end

A = collect(1.0:16.0)
linear_example!(CPU(), 4, size(A))(A)
A

# output

16-element Vector{Float64}:
 2.0
 3.0
 4.0
 5.0
 3.0
 4.0
 5.0
 6.0
 4.0
 5.0
 6.0
 7.0
 5.0
 6.0
 7.0
 8.0
```

Inside a kernel, [`@groupsize`](@ref) and [`@ndrange`](@ref) query the launch configuration:

```jldoctest
using KernelAbstractions

@kernel function scale!(A, factor)
    N = @uniform prod(@groupsize())
    I = @index(Global, Linear)
    lmem = @localmem Float32 (N,)
    i = @index(Local, Linear)
    lmem[i] = factor
    @synchronize()
    @inbounds A[I] = A[I] * lmem[i]
end

A = collect(1.0:16.0)
scale!(CPU(), 8, size(A))(A, 2)
A

# output

16-element Vector{Float64}:
  2.0
  4.0
  6.0
  8.0
 10.0
 12.0
 14.0
 16.0
 18.0
 20.0
 22.0
 24.0
 26.0
 28.0
 30.0
 32.0
```

## Local memory, synchronization, and private memory

[`@localmem`](@ref) declares storage shared by all work items in a workgroup. Only **static**
local memory is supported at the moment: the allocation size must be known at compile time
(for example `@localmem Int (32,)` or `@localmem Int (N,)` where `N = prod(@groupsize())` and
the workgroup size is fixed when the kernel is constructed). Reads and writes must be
separated by [`@synchronize`](@ref) if they are performed by different work items:

```jldoctest
using KernelAbstractions

@kernel function reverse_block!(A)
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    N = @uniform prod(@groupsize())
    buf = @localmem Int (N,)
    buf[i] = i
    @synchronize()
    @inbounds A[I] = buf[N - i + 1]
end

A = collect(1.0:16.0)
reverse_block!(CPU(), 8, size(A))(A)
A

# output

16-element Vector{Float64}:
 8.0
 7.0
 6.0
 5.0
 4.0
 3.0
 2.0
 1.0
 8.0
 7.0
 6.0
 5.0
 4.0
 3.0
 2.0
 1.0
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

Obtain the backend from an array with [`get_backend`](@ref) and always call [`synchronize`](@ref) before reading results on the host.
See the [Quickstart](@ref) for a full walkthrough and the Examples section of the manual for larger patterns.
