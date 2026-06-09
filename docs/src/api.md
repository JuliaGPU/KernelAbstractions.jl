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

### Backends and arrays

```@docs
Backend
GPU
CPU
POCLBackend
get_backend
KernelAbstractions.allocate
KernelAbstractions.zeros
KernelAbstractions.ones
KernelAbstractions.copyto!
KernelAbstractions.pagelock!
KernelAbstractions.unsafe_free!
KernelAbstractions.functional
KernelAbstractions.supports_unified
KernelAbstractions.supports_atomics
KernelAbstractions.supports_float64
```

### Devices and execution

```@docs
synchronize
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

These macros help inspect the generated kernel code. GPU LLVM reflection is only supported
on the CPU backend via [`@ka_code_llvm`](@ref).

```@docs
@ka_code_typed
@ka_code_llvm
```

## Internal

```@docs
KernelAbstractions.partition
KernelAbstractions.@context
KernelAbstractions.argconvert
```
