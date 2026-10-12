# Writing kernels

These kernel language constructs are intended to be used inside [`@kernel`](@ref) functions.
They are not valid in ordinary Julia code.

## Constant arguments

Kernel functions allow input arguments to be marked with the [`@Const`](@ref) macro. It informs
the compiler that the memory accessed through that argument will not be written to as part of
the kernel, and that it does not alias any other memory in the kernel. If you are used to CUDA C,
this is similar to `const restrict`.

```jldoctest
using KernelAbstractions

@kernel function saxpy!(a, @Const(X), Y)
    I = @index(Global)
    @inbounds Y[I] = a * X[I] + Y[I]
end

a = 2.0
X = collect(1.0:8.0)
Y = fill(1.0, 8)
backend = CPU()
saxpy!(backend, 8, size(Y))(a, X, Y)
synchronize(backend)
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
backend = CPU()
fill_diagonal!(backend, 4, size(A))(A, 42)
synchronize(backend)
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
    l = @index(Local, Linear)    # work item within workgroup
    @inbounds A[I] = g + l
end

A = collect(1.0:16.0)
backend = CPU()
linear_example!(backend, 4, size(A))(A)
synchronize(backend)
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
backend = CPU()
scale!(backend, 8, size(A))(A, 2)
synchronize(backend)
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
backend = CPU()
reverse_block!(backend, 8, size(A))(A)
synchronize(backend)
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

[`@private`](@ref) declares per-work-item storage that survives across [`@synchronize`](@ref)
statements, and [`@uniform`](@ref) evaluates an expression outside the work-item scope so it
can be reused across `@synchronize` statements. For scratch storage that does not need to
survive across `@synchronize`, an `MArray` can be used instead.

## [Reductions and scans](@id group_collectives)

[`@groupreduce`](@ref) combines a value from every work-item of the workgroup and returns the
result to all of them. [`@groupscan`](@ref) gives each work-item the result up to itself, in
the order of `@index(Local, Linear)`; it is inclusive by default, and exclusive with
`inclusive = false`. Both take an associative operator and its neutral element, and
`@groupreduce` also needs the operator to be commutative.

```julia
@kernel function group_sums!(out, @Const(x))
    i = @index(Global, Linear)
    s = @groupreduce(+, x[i], zero(eltype(out)); groupsize = 256)
    if @index(Local, Linear) == 1
        out[@index(Group, Linear)] = s
    end
end
```

Like [`@synchronize`](@ref), they are collectives: every work-item of the workgroup has to
reach them, not in a branch or loop that only some of them take, or after an early `return`.
Use them as a statement of their own, optionally assigning the result, as in
`res = @groupreduce(+, val, zero(T))`. Work-items that pad a partial workgroup take part as
well: they contribute the neutral element, without evaluating `val`.

Both go through local memory, whose size has to be known when the kernel is compiled: a
kernel without a static workgroup size passes `groupsize`, a constant upper bound on the
number of work-items per workgroup.

They work on every backend. Where the backend can shuffle the values, `@groupreduce` first
reduces each sub-group, and then combines those results through local memory.
KernelAbstractions picks this automatically. The order in which values are combined can
differ, so floating-point results may vary between backends and launch configurations.

## [Tiles](@id tiles)

A tile is a group of `N` consecutive work-items, by `@index(Local, Linear)`. Constructing a
kernel with a tile width, `f(backend; tile = N)`, divides its workgroups into tiles, and
[`@tile()`](@ref @tile) returns the work-item's tile in the kernel. The work-items of a tile
exchange values with operations such as [`tile_shfl`](@ref) and [`tile_reduce`](@ref), without
going through local memory. All work-items of a tile have to execute each tile operation
together, but different tiles may take different branches, loop a different number of times,
or `return` early.

```julia
@kernel function tile_maxima!(out, @Const(x))
    t = @tile()
    m = tile_reduce(t, max, x[@index(Global, Linear)])
    if t.lane == 1
        out[@index(Tile)] = m
    end
end

kernel = tile_maxima!(backend; tile = 16)
kernel(out, x; ndrange = 16 * length(out))
```

The workgroup size has to be a multiple of `N`, and every tile has to lie entirely inside or
entirely outside of the `ndrange`; see [`@tile`](@ref) for the details. The `ndrange` can have
several dimensions, as long as that holds. Here, each tile holds the 4 × 4 nodes of an
element, and each workgroup holds four elements:

```julia
@kernel function element_sums!(out, @Const(u))
    t = @tile()
    I = @index(Global, NTuple)
    out[I...] = tile_reduce(t, +, u[I...])
end
element_sums!(backend, (4, 4, 4); tile = 16)(out, u; ndrange = size(u))    # size(u) == (4, 4, nelements)
```

### Choosing the tile width

Tiles aren't declared in the kernel: pass the width when constructing it, `f(backend;
tile = 16)` or `f(backend, workgroupsize; tile = 16)`. A kernel that uses `@tile()` or
`@index(Tile)` needs one. [`tile_width(kernel)`](@ref tile_width) returns it on the host, and
`tile_width(t)` in the kernel, where it is a constant. Like a static workgroup size, the width
is compiled into the kernel; construct the kernel again for another width.

Tiles are built on the backend's sub-groups (warps, wavefronts or SIMD-groups). The width
has to be a power of two of at most the sub-group width `W`, and tiles narrower than `W` share
a sub-group, yet still take their own paths. Not every backend can do that:
[`supports_tiles(backend, N)`](@ref supports_tiles) says whether it supports tiles of `N`
work-items, and constructing a kernel with a width it doesn't support is an error. CUDA,
AMDGPU, Metal and oneAPI, and OpenCL.jl on Intel GPUs, support widths of up to 32. For tiles of
the full sub-group width, pass `tile = KernelInterface.sub_group_size(backend)`.

KernelAbstractions' CPU backend has no tiles: its sub-group operations involve the whole
workgroup, so tiles can't take their own paths. To support it, write a variant of the kernel
that uses local memory or the workgroup collectives instead:

```julia
kernel = supports_tiles(backend, 16) ? element_sums!(backend; tile = 16) :
    element_sums_localmem!(backend)
```

## Choosing how work-items communicate

| To | Use | Available |
|---|---|---|
| share data in any pattern within a workgroup | [`@localmem`](@ref) and [`@synchronize`](@ref) | on every backend |
| reduce or scan a value over the workgroup | [`@groupreduce`](@ref), [`@groupscan`](@ref) | on every backend |
| exchange values within small groups of work-items whose control flow may differ between groups, e.g. the nodes of an element | [tiles](@ref tiles) | where [`supports_tiles(backend, N)`](@ref supports_tiles) is `true` |
| program the hardware's sub-groups directly | a [KernelInterface](@ref kernelinterface) kernel | as its [capability queries](@ref ki_subgroups) report |

Use the collectives for reductions and scans over the workgroup, and local memory with
`@synchronize` for other patterns; both work everywhere. Tiles fit when values only need to
move within small groups that take their own paths. They avoid local memory, but that doesn't
guarantee a faster kernel: on some GPUs, the local-memory version is as fast or faster.
Compare both on the devices you care about, and keep the local-memory version for the backends
without tiles.

### Kernel languages

KernelInterface is the low-level API that backends implement, and `@kernel` is a language
built on it, which takes care of the work-items that pad a partial workgroup and of who takes
part in collectives. A kernel is either a `@kernel` or a kernel written against
[KernelInterface](@ref kernelinterface), never a mix: don't call KernelInterface's device
functions, such as its sub-group shuffles and votes, in a `@kernel`. In a `@kernel`, use the
collectives or tiles instead. Host code can use KernelInterface's queries, e.g. to pick a
kernel or a tile width.

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

# static workgroup size and ndrange — fewer runtime checks, specialized per size
kernel = my_kernel(backend, 32, size(A))
kernel(A)
```

### Index ranges

Each entry of `ndrange` is either an extent (indices `1:n`) or a range of indices, so a kernel
can iterate over a region whose indices do not start at 1. `ndrange` can also be given as a
single range or as a `CartesianIndices`:

```julia
# static ndrange over the indices -2:N+3 along x and 0:M+1 along y
kernel = my_kernel(backend, (16, 16), (-2:N+3, 0:M+1))
kernel(A)

# dynamic ndrange
kernel = my_kernel(backend, (16, 16))
kernel(A, ndrange=(-2:N+3, 0:M+1))
kernel(A, ndrange=CartesianIndices(A))   # e.g. for an OffsetArray
```

Inside the kernel `@index(Global, Cartesian)` and `@index(Global, NTuple)` return the shifted
indices, `@index(Global, Linear)` counts the indices of the region from 1 in column-major order,
and `@ndrange()` returns the extents.

Obtain the backend from an array with [`get_backend`](@ref) and always call [`synchronize`](@ref) before reading results on the host.
See the [Quickstart](@ref) for a full walkthrough and the Examples section of the manual for larger patterns.

## Loops without a kernel

A kernel whose body is a loop over the indices of an array needs none of the kernel language
beyond the index itself. [`foreach_index`](@ref) launches such a loop directly, with one work item
per index of `eachindex(y, x)`:

```julia
function scale!(y, x)
    foreach_index(y, x) do i
        @inbounds y[i] = 2 * x[i] + 1
    end
    return y
end
```

The body is an ordinary Julia function, and it receives the index a `for i in eachindex(y, x)`
loop would: a linear index if the arrays have `IndexLinear` style, a `CartesianIndex` otherwise.
Passing every array the body indexes makes sure the index is valid for each, and that they all
live on the same backend. The launch is asynchronous like any other, and bounds checks are not
elided.

To loop over other indices than those of an array, give the backend and the index space, a range
or a `CartesianIndices` like an `ndrange`. The body then receives these indices as they are,
e.g. those of the interior of a matrix:

```julia
function smooth!(B, A)
    interior = CartesianIndices((2:size(A, 1)-1, 2:size(A, 2)-1))
    foreach_index(get_backend(A), interior) do I
        i, j = Tuple(I)
        @inbounds B[I] = (A[i-1, j] + A[i+1, j] + A[i, j-1] + A[i, j+1]) / 4
    end
    return B
end
```

Because the body becomes a kernel, every value it captures must have a known type — which is why
the examples wrap the loop in a function, and why the body must not assign to a captured variable.
Write the kernel out with [`@kernel`](@ref) when it needs more than an index: workgroup-level
indices, local memory, or synchronization.
