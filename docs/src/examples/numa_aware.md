# NUMA-aware SAXPY

This example demonstrates how to define and run a SAXPY kernel (single-precision `Y[i] = a * X[i] + Y[i]`) such that it runs efficiently on a system with multiple memory domains ([NUMA](https://en.wikipedia.org/wiki/Non-uniform_memory_access)) using multithreading. (You likely will need to fine-tune the value of `N` on your system of interest if you care about the particular measurement.)

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

1) Pin your threads systematically to the available physical (or virtual) CPU-cores. [ThreadPinning.jl](https://github.com/carstenbauer/ThreadPinning.jl) is your friend.
2) Opt-out of Julia's dynamic task scheduling (especially task migration) by using `CPU(; static=true)` instead of `CPU()`.
3) Initialize your data in parallel(!). It is of utmost importance to use a parallel access pattern for initialization that is as similar as possible as the access pattern of your computational kernel. The reason for this is ["NUMA first-touch policy"](https://queue.acm.org/detail.cfm?id=2513149#:~:text=This%20is%20called%20the%20first,policy%20associated%20with%20a%20task.). `KernelAbstractions.zeros(backend, dtype, N)` is your friend.


**Demonstration:**

If above example is run with 128 Julia threads on a Noctua 2 compute node (128 physical cores distributed over two AMD Milan 7763 CPUs with 4 NUMA domains each), one may get the following numbers (comments for demonstration purposes):

```
Memory Bandwidth (GB/s): 145.64 # backend = CPU(), init = :parallel
Compute (GFLOP/s): 24.27

Memory Bandwidth (GB/s): 333.83 # backend = CPU(; static=true), init = :parallel
Compute (GFLOP/s): 55.64

Memory Bandwidth (GB/s): 32.74 # backend = CPU(), init = :serial
Compute (GFLOP/s): 5.46

Memory Bandwidth (GB/s): 32.46 # backend = CPU(; static=true), init = :serial
Compute (GFLOP/s): 5.41
```

The key observations are the following:

* Serial initialization leads to subpar performance (at least a factor of 4.5) independent of the chosen CPU backend. This is a manifestation of remark 3 above.
* The static CPU backend gives >2x better performance than the one based on the dynamic `Threads.@spawn`. This is a manifestation of remark 2 (and, in some sense, also 1) above.