# Quickstart

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

You can construct a kernel for a specific backend by calling the kernel function
with the first argument being the device of type `KA.Device`, the second argument being the size
of the workgroup and the third argument being a static `ndrange`. The second and
third argument are optional. After instantiating the kernel you can launch it by
calling the kernel object with the right arguments and some keyword arguments that
configure the specific launch.

```julia
A = ones(1024, 1024)
ev = mul2_kernel(CPU(), 16)(A, ndrange=size(A))
wait(ev)
all(A .== 2.0)
```
`mul_kernel(CPU(), 16)` creates a kernel for the host device `CPU()` with a static
workgroup size of `16` and a dynamic `ndrange`. This returns a kernel that is launched with the inputs
and the dynamic `ndrange` as a keyword argument via `kernel(A,
ndrange=size(A))`. All of this is reduced to one line of code with the kernel
eventually returning an event `ev`. All kernels are launched asynchronously,
thus event `ev` specifies the current state of the execution.
The [`wait`] blocks the *host* until the event `ev` has completed on the device. This implies
that no new kernels will be launched by the host on any device until the wait is completed.
## Launching kernel on the device

To launch the kernel on a CUDA supported GPU we just have to generate the kernel
for the device of type `CUDADevice` provided by `CUDAKernels` (see [backends](backends.md)).

```julia
using CUDAKernels # Required to access CUDADevice
A = CUDA.ones(1024, 1024)
ev = mul2_kernel(CUDADevice(), 16)(A, ndrange=size(A))
wait(ev)
all(A .== 2.0)
```

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
different stream as the KA kernels.  Launching `mul_kernel` may start before `A
.= 1.0` has completed. To prevent this, we add a device-wide dependency to the
kernel by adding `dependencies=Event(CUDADevice())`.
```julia
ev = mul_kernel(CUDADevice(), 16)(A, ndrange=size(A), dependencies=Event(CUDADevice()))
```
This device dependency requires all kernels on the device to be compeleted before this kernel is launched.
In the same vain, multiple events may be added to a wait.

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