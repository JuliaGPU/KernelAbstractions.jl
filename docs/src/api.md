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
synchronize
allocate
```

## Host language

```@docs
KernelAbstractions.zeros
KernelAbstractions.supports_unified
```

## Profiler integration

KernelAbstractions exposes a small interface that backend or profiler packages
can implement to forward named ranges to external tracing profilers such as
NVIDIA Nsight Systems or Intel VTune. The defaults are no-ops, so calls are
free when no integration is loaded.

```@docs
profiling_range_active
profiling_range_start
profiling_range_end
@profiling_range
```

## Internal

```@docs
KernelAbstractions.Kernel
KernelAbstractions.partition
KernelAbstractions.@context
```
