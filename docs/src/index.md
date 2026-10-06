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
This means a kernel with `ndrange=(32, 30)` might be executed on a static block of `(32,32)`. In order to block these extra indices,
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
@kernel unsafe_indices=true function localmem(A)
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

### 0.10
- KernelAbstractions requires Julia 1.10 or later.
- The `CPU` backend is an OpenCL backend: `CPU` is an alias of `POCLBackend`, which compiles
  kernels like the GPU backends do and runs them with [PoCL](https://portablecl.org) on
  PoCL's own threads, and no longer on Julia tasks. `CPU(; static=true)` has been removed,
  as there is no dynamic task scheduling to opt out of anymore.
- KernelAbstractions is built on [KernelInterface](@ref kernelinterface), which defines
  `Backend` and the host-side functions (`allocate`, `synchronize`, …) that
  KernelAbstractions re-exports. User code is unaffected, but backends have to implement
  KernelInterface, so KernelAbstractions 0.10 needs a release of the backend package that
  supports it; see the [notes for backend implementations](@ref implementations_notes).
- The Enzyme extension has been removed temporarily, and is planned to return in a later
  0.10 release. Code that differentiates kernels with Enzyme has to stay on
  KernelAbstractions 0.9 until then.
- [`KernelAbstractions.@spawn`](@ref) launches kernels from a task like `Threads.@spawn`
  does, and additionally orders them after the work the spawning task has queued, and can
  select the device the task uses; see the [Quickstart](@ref).
- `KernelAbstractions.isgpu` has been removed. Like for `GPU` below, query a capability
  instead.
- `ndrange` entries may be index ranges, given statically (`kernel(backend, workgroupsize, (-2:N+3, 0:M+1))`)
  or at launch (`ndrange=(-2:N+3, 0:M+1)`, a single range, or a `CartesianIndices`).
  `@index(Global, Cartesian)` and `@index(Global, NTuple)` return the shifted indices.
- `@ka_code_llvm` and `@ka_code_typed` have been removed. They reflected on the host-side
  lowering of a kernel, and `@ka_code_llvm` additionally rejected GPU backends, which since
  the CPU backend became an OpenCL backend meant every backend. Use
  `KernelAbstractions.@device_code_llvm` and `KernelAbstractions.@device_code_typed` instead,
  which report on the code a backend actually generates; see [Reflection](@ref).
- `KernelAbstractions.GPU` is deprecated: and backends should now subtype `KernelAbstractions.Backend` directly. `GPU` is now an alias of `Backend` and will be removed in a future release. Code dispatching on `::GPU` should dispatch on `::Backend`, on concrete
  backend types, or on a capability such as `KernelAbstractions.supports_float64`.
- An exception thrown in a kernel on the `CPU` backend is reported as a
  `KernelAbstractions.POCL.KernelException` when the kernel has completed, like on GPU
  backends, and no longer as the original exception (e.g., a `BoundsError`). Depending on
  the debug level (`julia -g`), the kernel prints which exception it threw, on which
  work-item, and with `-g2` a backtrace.
- The `CPU` backend runs kernels on as many threads as Julia was started with (`julia -t`),
  like the thread-based `CPU` backend of 0.9 did, and no longer on one thread per hardware
  thread. Set `JULIA_KA_CPU_THREADS` to use a different number of threads; see [`CPU`](@ref).
- `@private` arrays are a `KernelAbstractions.PrivateArray` in stack storage on every backend,
  instead of a `StaticArrays.MArray` on most GPU backends, which was heap-allocated when passed
  to a function that isn't inlined. It is still a `StaticArray`, but static-array arithmetic,
  slicing and fast whole-array reductions now need StaticArrays to be loaded, and `copy` is
  not supported: write `SVector(Tuple(p))` for an immutable copy. See [`@private`](@ref).
  Backends no longer need to implement `KernelAbstractions.Scratchpad`: it is an overlay in
  `GPUCompiler.SHARED_METHOD_TABLE`, which backends that override
  `GPUCompiler.method_table_view` have to include, e.g., by implementing
  `GPUCompiler.method_tables` instead.
- Launching a kernel on the `CPU` backend costs more than in 0.9: the kernel is handed to
  PoCL's threads and the launching task waits for them, which takes several microseconds
  per launch even for an empty kernel. Code that launches many small kernels pays this every
  time. For example, a kernel over a 16×16×16 range took about 2.5 times as long per launch as
  with 0.9, and one over a 4×4×4 range about 3 times as long.
- On the `CPU` backend, don't pass a fixed workgroup size such as `64`. Omit it
  (`kernel(CPU())` instead of `kernel(CPU(), 64)`), so that it is chosen for every launch. A
  one-dimensional workgroup size pads a multidimensional range: with a workgroup size of 64
  and a range of `(16, 16, 16)`, each of the 256 workgroups has 64 work-items, of which 16
  are inside the range. A 7-point stencil over that range took 24 µs per launch with a
  workgroup size of 64, and 8 µs with the chosen one. Over a 128×128×128 range the chosen
  size was still about 14% faster.
- The `CPU` backend checks in every kernel whether a work-item is inside the range, even when
  the workgroups cover the range exactly, so kernels with a cheap body can be about twice
  as slow as with 0.9 ([#845](https://github.com/JuliaGPU/KernelAbstractions.jl/issues/845)).
- `foreach_index(f, A)` runs `f` once per index of the array `A` without writing a kernel out,
  and `foreach_index(f, backend, indices)` once per index in a range or `CartesianIndices`.

## Semantic differences

### To CUDA.jl/AMDGPU.jl

1. The kernels are automatically bounds-checked against either the dynamic or statically
   provided `ndrange`.
2. Kernels implictly return `nothing`

## Contributing
Please file any bug reports through Github issues or fixes through a pull
request. Any heterogeneous hardware or code aficionados is welcome to join us on
our journey.
