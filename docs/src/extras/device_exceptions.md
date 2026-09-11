# Device-side exceptions

Kernels running on the `POCLBackend` report device-side exceptions to the host as
`KernelException`s. The faulting work-item exits, and the exception is raised at the next
synchronization point. Because POCL launches are synchronous, that is normally the launch
itself:

```julia-repl
julia> using KernelAbstractions

julia> @kernel function oob(a)
           i = @index(Global, Linear)
           a[i + 1] = 1.0f0
       end;

julia> a = KernelAbstractions.zeros(POCLBackend(), Float32, 1);

julia> oob(POCLBackend())(a; ndrange = 1)
ERROR: KernelException: A BoundsError was thrown on device cpu-... : Out-of-bounds array access
For more details, run Julia with `-g2`
```

The mailbox is reset before the host exception is thrown, so subsequent kernels can run
normally. Results from a failed kernel may be incomplete. Exception handling does not relax
OpenCL's requirements for convergent work-group barriers.

## Diagnostic detail

Kernels default to Julia's `-g` setting, which selects how much diagnostic information a
kernel records:

- `0`: records only that an exception occurred.
- `1` (the default): also records the type and reason for common runtime errors, including
  bounds errors, domain errors, overflow, and inexact conversions.
- `2`: also records the work-item's local id and work-group id (both 1-based), a name for
  other exceptions, and a device-side backtrace.

```julia
try
    oob(POCLBackend())(a; ndrange = 1)
catch exc
    @show exc.dev exc.name exc.reason
    @show exc.work_item exc.work_group exc.backtrace
end
```

Backtrace entries are `(function, file, line)` tuples. Fields unavailable at the selected
level contain empty strings, empty vectors, or zero coordinates. Names, reasons, function
names, and file paths are limited to 63, 191, 127, and 127 bytes respectively; backtraces
contain at most 16 frames. Level 2 adds more code to throwing paths and is intended for
debugging.

The level can also be set per kernel, through the `debug_level` keyword of the lower-level
`@opencl` and `KernelAbstractions.POCL.clfunction` entry points:

```julia
import KernelAbstractions.POCL: @opencl
@opencl debug_level = 2 (a -> (a[2] = 1.0f0; return))(a)
```

## Synchronization and memory

Each device in an OpenCL context has a host-visible exception mailbox, allocated on first
launch. Queues targeting that device share the mailbox, so checking one queue may wait for
and report a failure from another queue on the same device. `exc.dev` identifies that
device. Only one faulting work-item records detailed diagnostics until the host consumes
the report. Its launch identifier and coordinates distinguish it from other work-items and
later kernels.

POCL's CPU device shares the host address space, so the mailbox is an ordinary host
allocation that device code reaches by pointer. Mailboxes are retained for the lifetime of
the process.
