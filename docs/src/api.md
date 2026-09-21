# API

## [Kernel language](@id api_kernel_language)

```@docs
@kernel
@Const
@index
@localmem
@private
@synchronize
@print
@uniform
@groupsize
@ndrange
```

## Host language

!!! note
    The [`Backend`](@ref) type hierarchy and most of the host-side management
    functions below ([`get_backend`](@ref), [`allocate`](@ref KernelAbstractions.allocate),
    [`synchronize`](@ref), device selection, …) are defined in the
    [KernelInterface](@ref kernelinterface) sibling package and re-exported by
    `KernelAbstractions`, so `KernelAbstractions.allocate` and
    `KernelInterface.allocate` are the same function. User code can keep calling
    them through `KernelAbstractions` as before. Note that this only applies to
    KernelAbstractions 0.10 and later: KernelAbstractions 0.9 defines its own
    versions of these functions, which are distinct from the `KernelInterface`
    ones.

### [Backends and arrays](@id api_backends_arrays)

```@docs
Backend
GPU
CPU
POCLBackend
get_backend
Adapt.adapt_storage(::Backend, ::Any)
KernelAbstractions.allocate
KernelAbstractions.zeros
KernelAbstractions.ones
KernelAbstractions.copyto!
KernelAbstractions.pagelock!
KernelAbstractions.unsafe_free!
KernelAbstractions.functional
KernelAbstractions.versioninfo
KernelAbstractions.supports_unified
KernelAbstractions.supports_atomics
KernelAbstractions.supports_float64
```

### Devices and execution

```@docs
synchronize
KernelAbstractions.@spawn
KernelAbstractions.record_event
KernelAbstractions.wait_event
KernelAbstractions.device
KernelAbstractions.ndevices
KernelAbstractions.device!
KernelAbstractions.priority!
```

### Kernel handles

```@docs
KernelAbstractions.Kernel
KernelAbstractions.workgroupsize
KernelAbstractions.ndrange
KernelAbstractions.backend
```

## Reflection

To look at the code a backend actually generates, wrap a kernel launch in one of the
`@device_code_*` macros below. They work the same on the CPU backend and on GPU
backends, and they are public, but not exported, so you must call them qualified:

```julia
KernelAbstractions.@device_code_llvm mul2(backend, 64)(A, ndrange=length(A))
```

```@docs
KernelAbstractions.@device_code_lowered
KernelAbstractions.@device_code_typed
KernelAbstractions.@device_code_warntype
KernelAbstractions.@device_code_llvm
KernelAbstractions.@device_code_native
KernelAbstractions.@device_code
```

## Internal

The functionalities in this section are considered internal and not part of the public API contract.
They are only documented here for developers and contributors of `KernelAbstractions.jl`, but should not be used by end users (and if they do, they should expect breakage without notice).

```@docs
KernelAbstractions.partition
KernelAbstractions.@context
KernelAbstractions.argconvert
KernelAbstractions.NDIteration.DynamicSize
KernelAbstractions.NDIteration.StaticSize
KernelAbstractions.NDIteration.NDRange
KernelAbstractions.NDIteration.StaticOffset
KernelAbstractions.NDIteration.DynamicOffset
KernelAbstractions.NDIteration.extents
KernelAbstractions.NDIteration.offsets
KernelAbstractions.NDIteration.linear_index
```
