# Coming from CUDA.jl

This page maps the kernel language of KernelAbstractions to that of
[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl). A [`@kernel`](@ref) runs on `CUDABackend()`
like on any other backend.

## Terminology

| KernelAbstractions | CUDA |
|--------------------|------|
| work-item | thread |
| workgroup | thread block |
| sub-group | warp |
| `ndrange`, in work-items | grid, in blocks |
| local memory, per workgroup | shared memory |
| private memory, per work-item | registers and local memory |

Note that CUDA's local memory is private to a thread, while KernelAbstractions' local memory
is shared by the workgroup.

## Kernel language

| KernelAbstractions | CUDA.jl |
|--------------------|---------|
| `@index(Local, Linear)` | `threadIdx().x` in a 1-D block |
| `@index(Local, Cartesian)[2]` | `threadIdx().y` |
| `@index(Group, Linear)` | `blockIdx().x` in a 1-D grid |
| `@index(Group, Cartesian)[2]` | `blockIdx().y` |
| `@index(Global, Linear)` | `(blockIdx().x - 1) * blockDim().x + threadIdx().x` in 1-D |
| `@index(Global, Cartesian)[2]` | `(blockIdx().y - 1) * blockDim().y + threadIdx().y` |
| `@groupsize()[2]` | `blockDim().y` |
| `cld(@ndrange()[2], @groupsize()[2])` | `gridDim().y` |
| `@localmem T dims` | `CuStaticSharedArray(T, dims)` |
| `@private T dims` | a per-thread array, e.g. an `MArray` |
| `@synchronize()` | `sync_threads()` |
| `@Const(A)` | `CUDA.Const(A)` |
| `@print` | `@cuprint` |
| `Atomix.@atomic` | `CUDA.@atomic` |

The first, second and third dimension correspond to `x`, `y` and `z`, and the linear
indices of multi-dimensional workgroups and groups count `x` fastest, like CUDA numbers the
threads of a block. For an `ndrange` of [index ranges](@ref "Index ranges"),
`@index(Global, Cartesian)` returns the shifted indices, e.g. 3 for the first work-item of
`3:10`.

The right column is what a CUDA.jl kernel computes, not what CUDA.jl's functions return in a
`@kernel`: KernelAbstractions may launch a kernel on a 1-D grid, where `threadIdx().y` is
always 1, so don't call them in a `@kernel`.

Only static local memory is supported, so there is no equivalent of `CuDynamicSharedArray`.
Unlike `CUDA.@atomic`, `Atomix.@atomic A[i] = A[i] + x` is an atomic store of a value that
is computed separately. Write `Atomix.@atomic A[i] += x` for an atomic update.

## Launching kernels

```julia
using CUDA

function axpy!(y, a, x)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(y)
        @inbounds y[i] = a * x[i] + y[i]
    end
    return
end

@cuda threads=256 blocks=cld(length(y), 256) axpy!(y, a, x)
```

```julia
using KernelAbstractions

@kernel function axpy!(y, a, @Const(x))
    i = @index(Global, Linear)
    @inbounds y[i] = a * x[i] + y[i]
end

axpy!(get_backend(y), 256)(y, a, x, ndrange=length(y))
```

A kernel is given the number of work-items, its `ndrange`, rather than the number of blocks.
KernelAbstractions partitions it into workgroups, and only runs the kernel body on the
work-items inside the `ndrange`, so the kernel doesn't check the bounds itself, unless it
[opts out](@ref unsafe_indices) with `unsafe_indices = true`. The padding work-items of a
partial workgroup still reach every [`@synchronize`](@ref), so control flow around a
`@synchronize` has to compute the values it depends on with [`@uniform`](@ref), which also
evaluates them on the padding work-items. CUDA has no need for that: it runs the whole
kernel on every thread, and the kernel checks the bounds itself.

All kernels are launched asynchronously. [`synchronize`](@ref) waits for them, and calls
`CUDA.synchronize()` on `CUDABackend()`.
