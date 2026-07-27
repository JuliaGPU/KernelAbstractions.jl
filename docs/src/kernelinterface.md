# [KernelInterface](@id kernelinterface)

```@meta
CurrentModule = KernelInterface
```

`KernelInterface` (conventionally imported as `KI`) is the low-level API that
backends implement, and that `KernelAbstractions` builds its higher-level kernel
language on top of.

It ships as a standalone package under `lib/KernelInterface` with **no
dependencies outside the standard library**, so a backend can implement the
interface without taking on `KernelAbstractions` or its compiler stack:

```julia
using KernelInterface
const KI = KernelInterface
```

`KernelAbstractions` re-exports it, so `KernelAbstractions.KernelInterface` and
`KernelAbstractions.KI` refer to the same module.

!!! note
    Most of the functions below are stubs with no methods. They exist so that
    backends can add device-side implementations with
    `GPUCompiler.@device_override`, and so kernels can call them generically.
    Calling one without a backend that implements it is a `MethodError`.

```@docs
KernelInterface
```

## Device-side API

These are called from inside a kernel. A backend provides each one with

```julia
@device_override KI.get_global_id() = ...
```

along with the corresponding on-device functionality.

### Indexing

All index queries are **1-based** and return a named tuple of `x`, `y` and `z`
components.

```@docs
get_global_size
get_global_id
get_local_size
get_local_id
get_num_groups
get_group_id
```

### Sub-groups

```@docs
get_sub_group_size
get_max_sub_group_size
get_num_sub_groups
get_sub_group_id
get_sub_group_local_id
```

### Barriers

```@docs
barrier
sub_group_barrier
```

### Memory

```@docs
localmemory
```

### Communication

```@docs
shfl_down
shfl_down_types
```

### Printing

```@docs
KernelInterface._print
```

`_print` is the one device-side function with a working host fallback: it prints
its arguments with `Base.print`, unwrapping any `Val`-wrapped literals. That is
what makes [`KernelAbstractions.@print`](@ref) usable outside of a kernel.

## Host-side API

### Backend queries

```@docs
max_work_group_size
sub_group_size
multiprocessor_count
```

### Compilation and launching

```@docs
Kernel
kernel_function
kernel_max_work_group_size
check_launch_args
argconvert
KernelInterface.@kernel
```

!!! note
    `KI.@kernel` is **not** `KernelAbstractions.@kernel`. `KI.@kernel` wraps a
    backend's own compile-and-launch path — the equivalent of `@cuda` or
    `@metal` — and prefixes a *call*. [`KernelAbstractions.@kernel`](@ref)
    prefixes a *definition* and produces a kernel written in the higher-level
    KernelAbstractions language.

## Implementing a backend

A backend must, at minimum:

1. `@device_override` the device-side functions it supports. The indexing
   queries and [`barrier`](@ref) are required; sub-group and
   [`shfl_down`](@ref) support is optional.
2. Implement [`argconvert`](@ref) and [`kernel_function`](@ref) for its backend
   type, returning a [`Kernel`](@ref).
3. Make that `Kernel` callable, accepting `numworkgroups` and `workgroupsize` as
   a scalar `Integer` or a 1-, 2- or 3-element tuple. Use
   [`check_launch_args`](@ref) to validate them, or check them directly.
4. Report its limits through [`kernel_max_work_group_size`](@ref) and, where
   applicable, [`max_work_group_size`](@ref), [`sub_group_size`](@ref) and
   [`multiprocessor_count`](@ref).

The PoCL backend in `src/pocl/backend.jl` is a complete worked example.

See also the [notes for backend implementations](@ref implementations_notes).
