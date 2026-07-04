# Debugging POCL compilation

The `CPU` backend of KernelAbstractions.jl is implemented on top of
[PoCL](https://portablecl.org/) (`POCLBackend`). Kernels are compiled to SPIR-V
and then handed to PoCL, which JIT-compiles them for the host CPU using LLVM.
When a kernel fails to compile — or compiles but misbehaves — it can be useful
to see what PoCL itself is doing.

## Enabling PoCL's debug output

PoCL's runtime honors the `POCL_DEBUG` environment variable, which takes a
comma-separated list of message categories. A good starting point is to enable
errors and warnings:

```sh
POCL_DEBUG=err,warn julia --project my_script.jl
```

This prints PoCL's internal error and warning messages (e.g. from SPIR-V
ingestion, LLVM code generation, or kernel argument handling) to standard
error, with timestamps and the source location inside PoCL that emitted them:

```
[2026-07-04 22:05:12.317942887] PoCL: in fn compile_and_link_program at line 773:  *** INFO ***  |      LLVM |     BUILDING for device: cpu
```

Since PoCL reads the variable when the library is initialized, it can also be
set from within Julia, as long as this happens before the first kernel launch
(or any other use of the backend):

```julia
ENV["POCL_DEBUG"] = "err,warn"

using KernelAbstractions
# ... launch kernels as usual
```

When in doubt, prefer setting the variable in the shell before starting Julia.

## Other useful categories

Beyond `err` and `warn`, some categories that are helpful when debugging
compilation issues:

- `llvm`: messages from PoCL's LLVM-based kernel compiler, including the
  work-group vectorization and linking steps.
- `cache`: shows where PoCL caches compiled kernels on disk, and whether a
  kernel was recompiled or loaded from the cache.
- `general`: high-level runtime activity, such as device initialization and
  program builds.
- `all`: everything. Very verbose, but useful as a last resort.

For example:

```sh
POCL_DEBUG=err,warn,llvm julia --project my_script.jl
```

!!! note
    PoCL caches compiled kernels on disk (by default under
    `~/.cache/pocl/kcache`). If a compilation step you expect does not show up
    in the debug output, the kernel was likely loaded from the cache — the
    debug output reports this with messages such as
    `Found cached compiled SPIRV binary at ..., skipping compilation`.
    Set `POCL_KERNEL_CACHE=0` to disable the cache and force recompilation,
    or `POCL_CACHE_DIR` to relocate it.
