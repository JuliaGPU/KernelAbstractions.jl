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
`KernelAbstractions.KI` refer to the same module. This includes the
[`Backend`](@ref) type hierarchy and the host-side management API: these are
defined here in `KernelInterface`, and `KernelAbstractions.Backend`,
`KernelAbstractions.allocate`, `KernelAbstractions.synchronize` and so on are
the same objects, so user code keeps using them through `KernelAbstractions`
unchanged.

!!! compat "KernelAbstractions 0.10"
    This only holds for KernelAbstractions 0.10 and later. KernelAbstractions
    0.9 predates `KernelInterface` and defines its own `Backend`, `allocate`,
    `synchronize`, etc. — those are **different** functions and types from the
    `KernelInterface` ones. Methods added to one are not seen by the other, so
    a backend targeting both must implement both. KernelAbstractions 0.10 is
    based on KernelInterface, so any KernelInterface functionality does not need
    to be reimplemented for KernelAbstractions.

!!! note
    Most of the device-side functions below are stubs with no methods. They
    exist so that backends can add device-side implementations with
    `GPUCompiler.@device_override`, and so kernels can call them generically.
    Calling one without a backend that implements it is a `MethodError`.

```@docs
KernelInterface
```

## Backend hierarchy

Backends subtype [`Backend`](@ref), and everything else in the interface dispatches on
that type. It and the host-side management functions below are re-exported by
`KernelAbstractions`, so their canonical docstrings are on the
[API page](@ref api_backends_arrays).

```@docs; canonical=false
Backend
get_backend
```

## Device-side API

These are called from inside a kernel. A backend provides each one with

```julia
@device_override KI.get_local_id(::Type{T}) where {T} = ...
```

along with the corresponding on-device functionality.

### Indexing

All index queries are **1-based** and return a named tuple of `x`, `y` and `z`
components. They take an optional integer type `T` for the components, defaulting
to `Int`, so a kernel can request e.g. `Int32` indices with
`KI.get_global_id(Int32)`. The operands are converted to `T` before any arithmetic,
and the result is the exact value modulo `T`: a query never throws, and a value that
doesn't fit wraps around, as with `x % T`.

Backends implement the four primitive queries. [`get_global_id`](@ref) and
[`get_global_size`](@ref) have fallbacks derived from them, which backends with a native
builtin (e.g. SPIR-V and Metal) should override.

```@docs
get_local_id
get_group_id
get_local_size
get_num_groups
get_global_id
get_global_size
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

Several of these have generic fallbacks. Each docstring notes which methods
a backend **must** implement and which ones are optional.

### Memory

```@docs; canonical=false
allocate
KernelInterface.zeros
KernelInterface.ones
copyto!
pagelock!
unsafe_free!
```

### Execution

```@docs; canonical=false
synchronize
record_event
wait_event
priority!
```

### Device management

```@docs; canonical=false
device
ndevices
device!
```

### Capability queries

```@docs; canonical=false
functional
versioninfo
supports_unified
supports_atomics
supports_float64
```

### Limits

```@docs
max_work_group_size
launch_configuration
max_work_group_dims
max_num_groups
sub_group_size
multiprocessor_count
```

### Compilation and launching

```@docs
Kernel
kernel_function
argconvert
launch
KernelInterface.@launch
```

## Implementing a backend

A backend must, at minimum:

1. Define a backend type subtyping [`Backend`](@ref), and implement [`get_backend`](@ref)
   for its array type.
2. Implement the host-side management functions for that type:
   [`allocate`](@ref), [`copyto!`](@ref), [`synchronize`](@ref) and
   [`unsafe_free!`](@ref) are required; the remaining functions under
   [Host-side API](@ref) have fallbacks that only need overriding when the
   defaults don't apply.
3. Extend `Adapt.adapt_storage(::NewBackend, x)` so that
   [`adapt(backend, x)`](@ref Adapt.adapt_storage(::Backend, ::Any)) moves
   data to the backend, preferably by delegating to its array type:
   `Adapt.adapt_storage(::NewBackend, x) = adapt(NewArray, x)`.
4. `@device_override` the device-side functions it supports. The four primitive index
   queries and [`barrier`](@ref) are required; sub-group and [`shfl_down`](@ref) support
   is optional.
5. Implement [`argconvert`](@ref) and [`kernel_function`](@ref) for its backend
   type, returning a [`Kernel`](@ref).
6. Implement [`launch`](@ref), which receives an already validated `NTuple{3, Int}` of
   work-groups and of work-items. For CUDA.jl, that is
   ```julia
   KI.launch(k::KI.Kernel{CUDABackend}, groups::Dims{3}, items::Dims{3}, args::Vararg{Any, N}; kwargs...) where {N} =
       k.kern(args...; threads = items, blocks = groups, kwargs...)
   ```
7. Compute the typed index queries with `% T`, not `T(x)`: a checked conversion leaves
   an error branch in every kernel.
8. Report its limits through [`max_work_group_size`](@ref) (for the backend and for a
   kernel), [`max_work_group_dims`](@ref) and [`max_num_groups`](@ref), and where
   applicable [`sub_group_size`](@ref) and [`multiprocessor_count`](@ref). It may
   recommend work-group sizes with [`launch_configuration`](@ref).

The PoCL backend in `src/pocl/backend.jl` is a complete worked example.

See also the [notes for backend implementations](@ref implementations_notes).
