# Coming from CUDA.jl

This page maps the kernel language of KernelAbstractions to [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).
A KernelAbstractions kernel runs on `CUDABackend()` as well as on every other backend.

## Terminology

| KernelAbstractions | CUDA |
|--------------------|------|
| work-item | thread |
| workgroup | thread block |
| sub-group | warp |
| `ndrange`, counted in work-items | grid, counted in blocks |
| local memory, shared by a workgroup | shared memory |
| private memory, per work-item | registers and local memory |

CUDA's local memory is private to a thread, while KernelAbstractions' local memory is shared by a workgroup.

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
| `@Const(A)` in the signature | `Base.Experimental.Const(A)` |
| `@print` | `@cuprint` |
| `Atomix.@atomic` | `CUDA.@atomic` |

The first, second and third dimension correspond to `x`, `y` and `z`.
Linear indices of multi-dimensional workgroups and groups count `x` fastest, like CUDA numbers the threads of a block.
For an `ndrange` of [index ranges](@ref "Index ranges"), `@index(Global, Cartesian)` returns the indices of the ranges, e.g. 3 for the first work-item of `3:10`.
There is no equivalent of `CuDynamicSharedArray`: the size of [`@localmem`](@ref) has to be known when the kernel is compiled.

## Launching

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

axpy!(get_backend(y), 256)(y, a, x; ndrange = length(y))
```

A KernelAbstractions kernel is given the number of work-items, its `ndrange`, rather than the number of blocks.
KernelAbstractions computes the number of workgroups from it, and only runs the kernel body on the work-items inside the `ndrange`, so the kernel needs no bounds check, unless it is declared with `@kernel unsafe_indices=true`.
The padding work-items of a partial workgroup still reach every [`@synchronize`](@ref).
A value that control flow around a `@synchronize` depends on must therefore be computed with [`@uniform`](@ref), which also evaluates it on the padding work-items.
CUDA needs no equivalent, because it runs the whole kernel on every thread, and the kernel checks the bounds itself.

Launches are asynchronous on both.
Call [`synchronize`](@ref), which calls `CUDA.synchronize()` on `CUDABackend()`, before reading the results on the host.
