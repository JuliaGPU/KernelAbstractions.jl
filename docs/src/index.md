# KernelAbstractions

`KernelAbstractions.jl` is a package that allows you to write GPU-like kernels that
target different execution backends. It is intended to be a minimal, and performant
library that explores ways to best write heterogenous code.

!!! note
    While `KernelAbstraction.jl` is focused on performance portability, it is GPU-biased
    and therefore the kernel language has several constructs that are necessary for good
    performance on the GPU, but may hurt performance on the CPU.

## Quickstart

### Writing your first kernel

Kernel functions have to be marked with the [`@kernel`](@ref). Inside the `@kernel` macro
you can use the [kernel language](@ref api_kernel_language). As an example the `mul2` kernel
below will multiply each element of the array `A` by `2`. It uses the [`@index`](@ref) macro
to obtain the global linear index of the current workitem.

```julia
@kernel function mul2(A)
  I = @index(Global)
  A[I] = 2 * A[I]
end
```

### Launching your first kernel

You can construct a kernel for a specific backend by calling the kernel function
with the first argument being the device kind, the second argument being the size
of the workgroup and the third argument being a static `ndrange`. The second and
third argument are optional. After instantiating the kernel you can launch it by
calling the kernel object with the right arguments and some keyword arguments that
configure the specific launch. The example below creates a kernel with a static
workgroup size of `16` and a dynamic `ndrange`. Since the `ndrange` is dynamic it
has to be provided for the launch as a keyword argument.

```julia
A = ones(1024, 1024)
kernel = mul2(CPU(), 16)
event = kernel(A, ndrange=size(A))
wait(event)
all(A .== 2.0)
```

!!! danger
    All kernel launches are asynchronous, each kernel produces an event token that
    has to be waited upon, before reading or writing memory that was passed as an
    argument to the kernel. See [dependencies](@ref dependencies) for a full
    explanation.

If you have a GPU attached to your machine, it's equally as easy to launch your
kernel on it instead. For example, launching on a CUDA GPU:

```julia
using CUDAKernels # Required to access CUDADevice
A = CUDA.ones(1024, 1024)
kernel = mul2(get_computing_device(A), 16)
# ... the rest is the same!
```

AMDGPU (ROCm) support is also available via the ROCKernels.jl package, although
at this time it is considered experimental. Ping `@jpsamaroo` in any issues
specific to the ROCKernels backend.

## Important differences to Julia

1. Functions inside kernels are forcefully inlined, except when marked with `@noinline`.
2. Floating-point multiplication, addition, subtraction are marked contractable.

## Important differences to CUDA.jl/AMDGPU.jl

1. The kernels are automatically bounds-checked against either the dynamic or statically
   provided `ndrange`.
2. Functions like `Base.sin` are mapped to `CUDA.sin`/`AMDGPU.sin`.

## Important differences to GPUifyLoops.jl

1. `@scratch` has been renamed to `@private`, and the semantics have changed. Instead
   of denoting how many dimensions are implicit on the GPU, you only ever provide the
   explicit number of dimensions that you require. The implicit CPU dimensions are
   appended.

## How to debug kernels

*TODO*

## How to profile kernels

*TODO*
