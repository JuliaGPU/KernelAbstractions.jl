# NUMA-aware SAXPY

This example demonstrates how to define and run a SAXPY kernel (single-precision `Y[i] = a * X[i] + Y[i]`) such that it runs efficiently on a system with multiple memory domains ([NUMA](https://en.wikipedia.org/wiki/Non-uniform_memory_access)) with the multithreaded `CPU` backend. (You likely will need to fine-tune the value of `N` on your system of interest if you care about the particular measurement.)

````@eval
using Markdown
using KernelAbstractions
path = joinpath(dirname(pathof(KernelAbstractions)), "..", "examples/numa_aware.jl")
Markdown.parse("""
```julia
$(read(path, String))
```
""")
````

**Important remarks:**

The `CPU` backend runs kernels with [PoCL](https://portablecl.org) on PoCL's own threads, not on Julia's. That determines how to apply the usual advice for NUMA systems:

1) Pin the threads that run the kernels. Set PoCL's `POCL_AFFINITY=1` environment variable before the backend is first used. It pins each of PoCL's threads to one of the CPUs the process may run on (respecting e.g. `taskset`), a core of its own first, spread over the NUMA domains, so that fewer threads than cores still use the memory bandwidth of every domain. Tools that pin Julia's threads, like [ThreadPinning.jl](https://github.com/carstenbauer/ThreadPinning.jl), don't affect the threads that run the kernels. Without pinning, the operating system moves the threads around, and where they touch the memory first says little about where they run later.
2) Choose the number of threads with `julia -t N` or the `JULIA_KA_CPU_THREADS` environment variable; see [`CPU`](@ref).
3) Initialize your data in parallel(!), with a kernel. Under the ["NUMA first-touch policy"](https://queue.acm.org/detail.cfm?id=2513149#:~:text=This%20is%20called%20the%20first,policy%20associated%20with%20a%20task.) a page of memory is placed in the memory domain of the thread that first writes to it. `fill!` writes from the calling thread, which places all the memory in that thread's domain. The example instead allocates with `KernelAbstractions.allocate` and writes the initial values with a kernel (`init = :parallel`). PoCL gives each thread a contiguous block of a kernel's workgroups, so a later kernel with as many workgroups mostly works on each part of the array from the thread that touched it first. `KernelAbstractions.zeros(backend, dtype, N)` and `KernelAbstractions.ones` initialize with a kernel too, for arrays of bits types of at least 1 MiB; smaller arrays, and arrays of other types, are filled from the calling thread.

This needs KernelAbstractions 0.10.2 or later, which avoids the masked memory accesses that made kernels slow on some AMD CPUs, and the `pocl_standalone_jll` 7.2.3+1 or later that it installs, whose PoCL schedules workgroups and pins threads this way.

**Demonstration:**

On a system with two AMD EPYC 7502 (2 × 32 cores, 2 NUMA domains), the example (`N = 512_000_000`, workgroups of 1024) gets the following memory bandwidth, in GB/s:

| threads | initialization | `POCL_AFFINITY=1` | `POCL_AFFINITY` unset |
|--:|---|--:|--:|
| 64 | `init = :parallel`, or `zeros` and `ones` | 242 | 130–155 |
| 64 | `init = :serial` | 142 | 137 |
| 32 | `init = :parallel`, or `zeros` and `ones` | 157 | 121–135 |
| 32 | `init = :serial` | 124 | 121–139 |

A plain `Threads.@threads :static` loop with its threads pinned by ThreadPinning.jl gets 240 GB/s with 64 threads. Initializing serially puts all the memory in one domain. Without pinning, the threads that touch the memory first aren't reliably the ones that work on it later, and the numbers vary from run to run with where the operating system places the threads.

On a system with a single memory domain, pinning and the initialization don't change where memory is placed: on an AMD Ryzen 9 9950X with 8 or 16 threads, all configurations get 53–55 GB/s.
