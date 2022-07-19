# Quickstart

## Terminology
Because CUDA is the most popular GPU programming environment, we can use it as a
reference for defining terminology in KA. A *workgroup* is called a block in
NVIDIA CUDA and designates a group of threads acting in parallel, preferably
in lockstep. For the GPU, the workgroup size is typically around 256, while for the CPU,
it is usually less than or equal to the number of CPU cores. An *ndrange* is
called a grid in NVIDIA CUDA and designates the total number of work items. If
using a workgroup of size 1 (non-parallel execution), the ndrange is the
number of items to iterate over in a loop.

## Writing your first kernel

Kernel functions are marked with the [`@kernel`](@ref). Inside the `@kernel` macro
you can use the [kernel language](@ref api_kernel_language). As an example, the `mul2`` kernel
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
`mul2_kernel(CPU(), 16)`. The first argument is a device of type `KA.Device`,
the second argument being the workgroup size. This returns a generated kernel
executable that is then executed with the input argument `A` and the additional
argument being a static `ndrange`.
```julia
A = ones(1024, 1024)
ev = mul2_kernel(CPU(), 16)(A, ndrange=size(A))
wait(ev)
all(A .== 2.0)
```
The kernel eventually returns an event `ev`. All kernels are launched
asynchronously with the event `ev` specifies the current state of the execution.
The [`wait`] blocks the *host* until the event `ev` has completed on the device.
This implies that the host will launch no new kernels on any device until the
wait returns.
## Launching kernel on the device

To launch the kernel on a backend-supported device `isa(device, KA.GPU)` (e.g., `CUDADevice()`, `ROCDevice()`, `oneDevice()`), we generate the kernel
for this device provided by `CUDAKernels`, `ROCKernels`, or `oneAPIKernels`.

First, we initialize the array using the Array constructor of the chosen device with

```julia
using CUDAKernels # Required to access CUDADevice
A = CuArray(ones(1024, 1024))
```

```julia
using ROCKernels # Required to access CUDADevice
A = ROCArray(ones(1024, 1024))
```

```julia
using oneAPIKernels # Required to access CUDADevice
A = oneArray(ones(1024, 1024))
```
The kernel generation and execution are then
```julia
ev = mul2_kernel(device, 16)(A, ndrange=size(A))
wait(ev)
all(A .== 2.0)
```

For simplicity, we stick with the case of `device=CUDADevice()`.

## Synchronization
!!! danger
    All kernel launches are asynchronous, each kernel produces an event token that
    has to be waited upon, before reading or writing memory that was passed as an
    argument to the kernel. See [dependencies](@ref dependencies) for a full
    explanation.

The code around KA may heavily rely on
[`GPUArrays`](https://github.com/JuliaGPU/GPUArrays.jl), for example, to
intialize variables.
```julia
using CUDAKernels # Required to access CUDADevice
function mymul(A::CuArray)
    A .= 1.0
    ev = mul2_kernel(CUDADevice(), 16)(A, ndrange=size(A))
    wait(ev)
    all(A .== 2.0)
end
```

These statement-level generated kernels like `A .= 1.0` are executed on a
different stream than the KA kernels.  Launching `mul_kernel` may start before `A
.= 1.0` has completed. To prevent this, we add a device-wide dependency to the
kernel by adding `dependencies=Event(CUDADevice())`.
```julia
ev = mul_kernel(CUDADevice(), 16)(A, ndrange=size(A), dependencies=Event(CUDADevice()))
```
This device dependency requires all kernels on the device to be completed before this kernel is launched.
In the same vein, multiple events may be added to a wait.

```julia
using CUDAKernels # Required to access CUDADevice
function mymul(A::CuArray, B::CuArray)
    A .= 1.0
    B .= 3.0
    evA = mul2_kernel(CUDADevice(), 16)(A, ndrange=size(A), dependencies=Event(CUDADevice()))
    evB = mul2_kernel(CUDADevice(), 16)(A, ndrange=size(A), dependencies=Event(CUDADevice()))
    wait(evA, evB)
    all(A .+ B .== 8.0)
end
```