# Quickstart

## Terminology
Because CUDA is the most popular GPU programming environment, we can use it as a
reference for defining terminology in KA. A *workgroup* is called a block in
NVIDIA CUDA and designates a group of threads acting in parallel, preferably
in lockstep. For the GPU, the workgroup size is typically around 256, while for the CPU,
it is usually a multiple of the natural vector-width. An *ndrange* is
called a grid in NVIDIA CUDA and designates the total number of work items. If
using a workgroup of size 1 (non-parallel execution), the ndrange is the
number of items to iterate over in a loop.

## Writing your first kernel

Kernel functions are marked with the [`@kernel`](@ref). Inside the `@kernel` macro
you can use the [kernel language](@ref api_kernel_language). As an example, the `mul2` kernel
below will multiply each element of the array `A` by `2`. It uses the [`@index`](@ref) macro
to obtain the global linear index of the current work item.

```julia
@kernel function mul2_kernel(A)
  I = @index(Global)
  A[I] = 2 * A[I]
end
```

## Launching kernel on the host

You can construct a kernel for a specific backend by calling the kernel with
`mul2_kernel(CPU(), 16)`. The first argument is a backend of type `KA.Backend`,
the second argument being the workgroup size. This returns a generated kernel
executable that is then executed with the input argument `A` and the additional
argument being a static `ndrange`.

```julia
dev = CPU()
A = ones(1024, 1024)
ev = mul2_kernel(dev, 64)(A, ndrange=size(A))
synchronize(dev)
all(A .== 2.0)
```

All kernels are launched asynchronously.
The [`synchronize`](@ref) blocks the *host* until the kernel has completed on the backend.

## Launching kernel on the backend

To launch the kernel on a backend-supported backend `isa(backend, KA.GPU)` (e.g., `CUDABackend()`, `ROCBackend()`, `oneAPIBackend()`), we generate the kernel
for this backend.

First, we initialize the array using the Array constructor of the chosen backend with

```julia
using CUDA: CuArray
A = CuArray(ones(1024, 1024))
```

```julia
using AMDGPU: ROCArray
A = ROCArray(ones(1024, 1024))
```

```julia
using oneAPI: oneArray
A = oneArray(ones(1024, 1024))
```
The kernel generation and execution are then
```julia
backend = get_backend(A)
mul2_kernel(backend, 64)(A, ndrange=size(A))
synchronize(backend)
all(A .== 2.0)
```

## Synchronization
!!! danger
    All kernel launches are asynchronous, use [`synchronize(backend)`](@ref)
    to wait on a series of kernel launches.

The code around KA may heavily rely on
[`GPUArrays`](https://github.com/JuliaGPU/GPUArrays.jl), for example, to
intialize variables.
```julia
function mymul(A)
    A .= 1.0
    backend = get_backend(A)
    ev = mul2_kernel(backend, 64)(A, ndrange=size(A))
    synchronize(backend)
    all(A .== 2.0)
end
```

```julia
function mymul(A, B)
    A .= 1.0
    B .= 3.0
    backend = get_backend(A)
    @assert get_backend(B) == backend
    mul2_kernel(backend, 64)(A, ndrange=size(A))
    mul2_kernel(backend, 64)(B, ndrange=size(B))
    synchronize(backend)
    all(A .+ B .== 8.0)
end
```

## Using task programming to launch kernels in parallel.

TODO
