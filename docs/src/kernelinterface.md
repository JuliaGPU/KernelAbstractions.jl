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

## Semantics

A few rules hold throughout the interface:

- **Execution is task-local.** A backend value (e.g. `CUDABackend()`) identifies a
  backend and its configuration, such as compiler options. Each Julia task has an active
  device per backend (selected with [`device!`](@ref)) and a queue on it. Host-side
  queries and compilation use the active device, allocations go to it, and copies and
  launches go to the calling task's queue. [`synchronize`](@ref) waits for that queue,
  and [`record_event`](@ref)/[`wait_event`](@ref) order work across queues. Switching
  devices doesn't synchronize.
- **Compiled kernels belong to a device.** Queries on a [`Kernel`](@ref)
  ([`max_work_group_size`](@ref), [`launch_configuration`](@ref)) answer for the device it
  was compiled for. Launching it after switching to another device either works or
  throws, but never runs on the wrong device.
- **Indices are 1-based**, and `x` is the fastest-varying dimension.
- **Capabilities default to "unsupported".** A backend that doesn't implement a
  `supports_*` query never claims support.

## Contract

What a backend implements, at a glance. The docstrings below have the details.

| | Required | Optional (fallback) |
|---|---|---|
| **Backend** | subtype [`Backend`](@ref); [`get_backend`](@ref) for its array type | |
| **Memory** | [`allocate`](@ref), [`copyto!`](@ref) | `allocate(...; unified=true)` (throws), [`pagelock!`](@ref) (`missing`), [`unsafe_free!`](@ref) (no-op) |
| **Execution** | [`synchronize`](@ref) (cooperative) | [`record_event`](@ref)/[`wait_event`](@ref) (synchronize), [`priority!`](@ref) (no-op) |
| **Devices** | with more than one device: [`ndevices`](@ref), [`device`](@ref), [`device!`](@ref), `device(backend, A)` | all four (a single device) |
| **Queries** | [`max_work_group_size`](@ref) (for the backend and for a kernel), [`max_work_group_dims`](@ref), [`max_num_groups`](@ref) | [`launch_configuration`](@ref) (the limit), [`multiprocessor_count`](@ref) (0), [`functional`](@ref) (`missing`), [`versioninfo`](@ref) |
| **Capabilities** | | [`supports_float64`](@ref), [`supports_atomics`](@ref), [`supports_unified`](@ref), [`supports_subgroups`](@ref), [`supports_shuffle`](@ref) (all `false`) |
| **Compilation** | [`argconvert`](@ref), [`kernel_function`](@ref), [`launch`](@ref) | |
| **Device** | [`get_local_id`](@ref), [`get_group_id`](@ref), [`get_local_size`](@ref), [`get_num_groups`](@ref), [`localmemory`](@ref), [`barrier`](@ref) | [`get_global_id`](@ref), [`get_global_size`](@ref) (derived from the primitive queries), [`_print`](@ref KernelInterface._print) (host `print`) |
| **Sub-groups** | if `supports_subgroups`: [`sub_group_size`](@ref), the sub-group queries, [`sub_group_barrier`](@ref); if `supports_shuffle(backend, T)`: [`shfl_down`](@ref) for `T` | |

Everything else, such as [`zeros`](@ref KernelInterface.zeros), [`ones`](@ref KernelInterface.ones),
the launch-keyword handling of [`Kernel`](@ref) and [`@launch`](@ref KernelInterface.@launch),
is generic and not meant to be overridden.

### Versioning

- Required methods only change in breaking releases (0.x → 0.x+1).
- Optional methods can be added in any release, with a fallback that is conservative:
  never claiming support, never wrong. Tests for them pass on the fallback, or are gated
  on a capability query.
- A patch release may add tests of behavior that was already specified; tests for newly
  specified behavior are new obligations and wait for a breaking release.

Backends test themselves against the contract with the testsuite in
`lib/KernelInterface/test`:

```julia
import KernelInterface
using Test
include(joinpath(pkgdir(KernelInterface), "test", "testsuite.jl"))
Testsuite.testsuite(MyBackend(), MyArray)
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

Sub-groups are optional ([`supports_subgroups`](@ref)). A work-group is divided into
sub-groups of at most [`sub_group_size(backend)`](@ref sub_group_size) work-items. Which
work-items form a sub-group, how many sub-groups there are, and which of them are partial
is unspecified, and differs between devices and work-group shapes. For example, CUDA forms
warps from consecutive linear work-item indices, while Intel's CPU OpenCL runtime forms
sub-groups per row of a multi-dimensional work-group, so that a 33×2 work-group consists of
four sub-groups of 32 and 1 work-items. What KernelInterface guarantees, and backends that
report sub-group support have to ensure:

- every work-item has a unique `(get_sub_group_id(), get_sub_group_local_id())` pair in its
  work-group, which doesn't change during the kernel;
- the sub-group ids are `1:get_num_sub_groups()`, and the lanes of a sub-group are
  `1:get_sub_group_size()`;
- a 1-D work-group of at most `sub_group_size(backend)` work-items is a single sub-group.

In particular, [`get_num_sub_groups`](@ref) can be larger than
`cld(prod(get_local_size()), get_max_sub_group_size())`. Storage for a value per sub-group
has to be sized for up to one sub-group per work-item, and code combining those values has
to use `get_num_sub_groups()` rather than compute the count.

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
```

### Printing

```@docs
KernelInterface._print
```

`_print` is the one device-side function with a working host fallback: it prints
its arguments with `Base.print`, unwrapping any `Val`-wrapped literals. That is
what makes [`KernelAbstractions.@print`](@ref) usable outside of a kernel.

## Host-side API

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

```@docs
supports_subgroups
supports_shuffle
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

A backend implements the required methods from the [contract](@ref Contract), and those
optional methods where it can do better than the fallback. In particular:

1. Define a backend type subtyping [`Backend`](@ref), and implement [`get_backend`](@ref)
   for its array type.
2. Extend `Adapt.adapt_storage(::NewBackend, x)` so that
   [`adapt(backend, x)`](@ref Adapt.adapt_storage(::Backend, ::Any)) moves
   data to the backend, preferably by delegating to its array type:
   `Adapt.adapt_storage(::NewBackend, x) = adapt(NewArray, x)`.
3. Implement [`kernel_function`](@ref), which receives the unconverted callable, and
   returns a [`Kernel`](@ref) that holds the backend value it was given and keeps that
   callable alive. Also implement [`launch`](@ref), which receives an already validated
   `NTuple{3, Int}` of work-groups and of work-items, and the arguments as a tuple. Pass
   that tuple on to the native launcher rather than splatting it: Julia doesn't turn a
   splat of more than 32 elements into a direct call, so kernels with many arguments would
   be slow to launch. For the PoCL backend, whose kernels hold the compiled kernel and the
   callable, `launch` is
   ```julia
   function KI.launch(k::KI.Kernel{POCLBackend}, groups::Dims{3}, items::Dims{3}, args::Tuple)
       f = k.kern.f
       GC.@preserve f POCL.launch_and_wait(
           k.kern.kernel, args; local_size = items, global_size = groups .* items
       )
       return nothing
   end
   ```
4. Compute the typed index queries with `% T`, not `T(x)`: a checked conversion leaves
   an error branch in every kernel.

The PoCL backend in `src/pocl/backend.jl` is a complete worked example.

See also the [notes for backend implementations](@ref implementations_notes).
