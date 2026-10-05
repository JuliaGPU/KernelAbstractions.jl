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

1) Pin the threads that run the kernels. Set PoCL's `POCL_AFFINITY=1` environment variable before the backend is first used, which pins each of PoCL's threads to a core. Tools that pin Julia's threads, like [ThreadPinning.jl](https://github.com/carstenbauer/ThreadPinning.jl), don't affect the threads that run the kernels.
2) Choose the number of threads with `julia -t N` or the `JULIA_KA_CPU_THREADS` environment variable; see [`CPU`](@ref).
3) Initialize your data in parallel(!), with a kernel. Under the ["NUMA first-touch policy"](https://queue.acm.org/detail.cfm?id=2513149#:~:text=This%20is%20called%20the%20first,policy%20associated%20with%20a%20task.) a page of memory is placed in the memory domain of the thread that first writes to it. `KernelAbstractions.zeros(backend, dtype, N)` and `fill!` write from the calling thread, which places all the memory in that thread's domain. The example instead allocates with `KernelAbstractions.allocate` and writes the initial values with a kernel (`init = :parallel`), so that the threads that run the computational kernel touch the memory first.


**Demonstration:**

So far, the example has only been measured with KernelAbstractions 0.10 on a system with a single memory domain, where pinning and the initialization don't change where memory is placed. With 16 Julia threads on an AMD Ryzen 9 5950X (16 physical cores, 1 NUMA domain), one gets the following numbers (comments for demonstration purposes):

```
Memory Bandwidth (GB/s): 41.19 # POCL_AFFINITY=1, init = :parallel
Compute (GFLOP/s): 6.87

Memory Bandwidth (GB/s): 41.22 # POCL_AFFINITY=1, init = :serial
Compute (GFLOP/s): 6.87

Memory Bandwidth (GB/s): 41.24 # POCL_AFFINITY unset, init = :parallel
Compute (GFLOP/s): 6.87

Memory Bandwidth (GB/s): 41.16 # POCL_AFFINITY unset, init = :serial
Compute (GFLOP/s): 6.86
```

As expected for a single memory domain, the four configurations don't differ: the kernel is limited by the memory bandwidth of the system. How much pinning and parallel initialization gain on a system with multiple memory domains has not been measured with the PoCL-based `CPU` backend yet.
