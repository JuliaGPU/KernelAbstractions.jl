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
[`GPUCompiler.jl`](https://github.com/JuliaGPU/GPUCompiler.jl).

### CUDA
```julia
import CUDA
using KernelAbstractions
```
[`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl) is currently the most mature way to program for GPUs.
This provides a backend `CUDABackend <: KA.Backend` to CUDA.

## Changelog

### 0.9
Major refactor of KernelAbstractions. In particular:
- Removal of the event system. Kernel are now implicitly ordered.
- Removal of backend packages, backends are now directly provided by CUDA.jl and similar

## Semantic differences

### To CUDA.jl/AMDGPU.jl

1. The kernels are automatically bounds-checked against either the dynamic or statically
   provided `ndrange`.
2. Kernels implictly return `nothing`

## Contributing
Please file any bug reports through Github issues or fixes through a pull
request. Any heterogeneous hardware or code aficionados is welcome to join us on
our journey.
