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

#### 0.9.5
- adds `@kernel cpu=false` 

#### 0.9.11
- adds `@kernel inbounds=true`

#### 0.9.22
- adds `KA.functional(::Backend)`

#### 0.9.32
- clarifies the semantics of `KA.copyto!` and adds `KA.pagelock!`
- adds support for multiple devices per backend

#### 0.9.34
Restricts the semantics of `@synchronize` to require convergent execution.
The OpenCL backend had several miss-compilations due to divergent execution of `@synchronize`.
The `CPU` backend always had this limitation and upon investigation the CUDA backend similarly requires convergent execution,
but allows for a wider set of valid kernels.

This highlighted a design flaw in KernelAbstractions. Most GPU implementations execute KernelAbstraction workgroups on static blocks
This means a kernel with `ndrange=(32, 30)` might be executed on a static block of `(32,32)`. In order to block these extra indicies,
KernelAbstraction would insert a dynamic boundscheck.

Prior to v0.9.34 a kernel like

```julia
@kernel function localmem(A)
    N = @uniform prod(@groupsize())
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    lmem = @localmem Int (N,) # Ok iff groupsize is static
    lmem[i] = i
    @synchronize
    A[I] = lmem[N - i + 1]
end
```

was lowered to GPU backends like this:

```julia
function localmem_gpu(A)
    if __validindex(__ctx__)
        N = @uniform prod(@groupsize())
        I = @index(Global, Linear)
        i = @index(Local, Linear)
        lmem = @localmem Int (N,) # Ok iff groupsize is static
        lmem[i] = i
        @synchronize
        A[I] = lmem[N - i + 1]
    end
end
```

This would cause an implicit divergent execution of `@synchronize`. 

With this release the lowering has been changed to:

```julia
function localmem_gpu(A)
    __valid_lane__ __validindex(__ctx__)
    N = @uniform prod(@groupsize())
    lmem = @localmem Int (N,) # Ok iff groupsize is static
    if __valid_lane__
        I = @index(Global, Linear)
        i = @index(Local, Linear)
        lmem[i] = i
    end
    @synchronize
    if __valid_lane__
        A[I] = lmem[N - i + 1]
    end
end
```

Note that this follow the CPU lowering with respect to `@uniform`, `@private`, `@localmem` and `@synchronize`.

Since this transformation can be disruptive, user can now opt out of the implicit bounds-check,
but users must avoid the use of `@index(Global)` and instead use their own derivation based on `@index(Group)` and `@index(Local)`.

```julia
@kernel unsafe_indicies=true function localmem(A)
    N = @uniform prod(@groupsize())
    gI = @index(Group, Linear)
    i = @index(Local, Linear)
    lmem = @localmem Int (N,) # Ok iff groupsize is static
    lmem[i] = i
    @synchronize
    I = (gI - 1) * N + i
    if i <= N && I <= length(A)
        A[I] = lmem[N - i + 1]
    end
end
```

## Semantic differences

### To CUDA.jl/AMDGPU.jl

1. The kernels are automatically bounds-checked against either the dynamic or statically
   provided `ndrange`.
2. Kernels implictly return `nothing`

## Contributing
Please file any bug reports through Github issues or fixes through a pull
request. Any heterogeneous hardware or code aficionados is welcome to join us on
our journey.
