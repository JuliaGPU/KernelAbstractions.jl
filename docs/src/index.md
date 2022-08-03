# KernelAbstractions

[`KernelAbstractions.jl`](https://github.com/JuliaGPU/KernelAbstractions.jl) (KA) is
a package that allows you to write GPU-like kernels targetting different
execution backends. KA intends to be a minimal and
performant
library that explores ways to write heterogeneous code. Although parts of
the package are still experimental, it has been used successfully as part of the
[Exascale Computing Project](https://www.exascaleproject.org/) to run Julia code
on pre-[Frontier](https://www.olcf.ornl.gov/frontier/) and
pre-[Aurora](https://www.alcf.anl.gov/aurora)
systems. Currently, profiling and debugging require backend-specific calls like, for example, in
[`CUDA.jl`](https://cuda.juliagpu.org/dev/development/profiling/).

!!! note
While KernelAbstraction.jl is focused on performance portability, it emulates GPU semantics and therefore the kernel language has several constructs that are necessary for good performance on the GPU, but serve no purpose on the CPU.
In these cases, we either ignore such statements entirely (such as with `@synchronize`) or swap out the construct for something similar on the CPU (such as using an `MVector`  to replace `@localmem`).
This means that CPU performance will still be fast, but might be performing extra work to provide a consistent programming model across GPU and CPU

## Supported backends
All supported backends rely on their respective Julia interface to the compiler
backend and depend on
[`GPUArrays.jl`](https://github.com/JuliaGPU/GPUArrays.jl) and
[`GPUCompiler.jl`](https://github.com/JuliaGPU/GPUCompiler.jl). KA provides
interface kernel generation modules to those packages in
[`/lib`](https://github.com/JuliaGPU/KernelAbstractions.jl/tree/master/lib).
After loading the kernel packages, KA will provide a `KA.Device` for that
backend to be used in the kernel generation stage.
### CUDA
```julia
using CUDA
using KernelAbstractions
using CUDAKernels
```
[`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl) is currently the most mature way to program for GPUs.
This provides a device `CUDADevice <: KA.Device` to
### AMDGPU
```julia
using AMDGPU
using KernelAbstractions
using ROCKernels
```
Experimental AMDGPU (ROCm) support is available via the
[`AMDGPU.jl`](https://github.com/JuliaGPU/AMDGPU.jl) and `ROCKernels.jl`. It
provides the device `ROCDevice <: KA.Device`. Please get in touch with `@jpsamaroo` for
any issues specific to the ROCKernels backend.
###  oneAPI
Experimental support for Intel GPUs has also been added through the oneAPI Intel
Compute Runtime interfaced to in
[`oneAPI.jl`](https://github.com/JuliaGPU/oneAPI.jl)

## Semantic differences

### To Julia

1. Functions inside kernels are forcefully inlined, except when marked with `@noinline`.
2. Floating-point multiplication, addition, subtraction are marked contractable.

### To CUDA.jl/AMDGPU.jl

1. The kernels are automatically bounds-checked against either the dynamic or statically
   provided `ndrange`.
2. Functions like `Base.sin` are mapped to `CUDA.sin`/`AMDGPU.sin`.

### To GPUifyLoops.jl

1. `@scratch` has been renamed to `@private`, and the semantics have changed. Instead
   of denoting how many dimensions are implicit on the GPU, you only ever provide the
   explicit number of dimensions that you require. The implicit CPU dimensions are
   appended.

## Contributing
Please file any bug reports through Github issues or fixes through a pull
request. Any heterogeneous hardware or code aficionados is welcome to join us on
our journey.
