# KernelInterface.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliagpu.github.io/KernelAbstractions.jl/dev/kernelinterface/)

KernelInterface (or `KI`) defines the low-level API that backends implement to
provide device- and host-side functionality for
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

KernelInterface focuses on the lower-level functionality shared amongst
backends such as kernel launching, device intrinsics, and host-side
operations such as allocation and synchronization.

Backends implement a small set of required methods (allocation, copies,
synchronization, compilation, a `launch` method and the primitive device
queries) and may override optional ones whose fallbacks are conservative; see
the [contract table](https://juliagpu.github.io/KernelAbstractions.jl/dev/kernelinterface/#Contract)
in the documentation. The testsuite in `test/testsuite.jl` checks it:

```julia
import KernelInterface
using Test
include(joinpath(pkgdir(KernelInterface), "test", "testsuite.jl"))
Testsuite.testsuite(MyBackend(), MyArray)
```


## Versioning

- Required methods only change in breaking releases (0.x → 0.x+1).
- Optional methods can be added in any release, with a conservative fallback
  (never claiming support, never wrong); tests for them pass on the fallback or
  are gated on a capability query.
- A patch release may add tests of behavior that was already specified; tests
  for newly specified behavior are new obligations and wait for a breaking
  release.


## License

KernelInterface is licensed under the [MIT license](LICENSE.md).
