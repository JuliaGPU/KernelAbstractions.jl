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

The `CPU` backend runs kernels on Julia's threads (see [`CPU`](@ref)). For NUMA locality:

1) Pin Julia's threads before the first launch with [ThreadPinning.jl](https://github.com/carstenbauer/ThreadPinning.jl). `pinthreads(:numa)` spreads threads across NUMA domains; `pinthreads(:cores)` follows the machine's topology. Both pin only the default pool. Start Julia with `-t N,0` or pass `threadpool = :all` to pin small launches on the launching thread too. Without pinning, threads can migrate away from the memory they first touched.
2) Choose the number of threads with `julia -t N` or the `JULIA_KA_CPU_THREADS` environment variable; see [`CPU`](@ref).
3) Initialize data with a parallel kernel. Under the [NUMA first-touch policy](https://queue.acm.org/detail.cfm?id=2513149#:~:text=This%20is%20called%20the%20first,policy%20associated%20with%20a%20task.), a memory page belongs to the domain of the thread that first writes it. `fill!` places pages in the calling thread's domain. The example uses `KernelAbstractions.allocate` followed by an initialization kernel (`init = :parallel`). Launches with as many workgroups start each block of them on the same thread, as `Threads.@threads :static` does, so later kernels mostly work on the data their thread touched first. `KernelAbstractions.zeros(backend, dtype, N)` and `KernelAbstractions.ones` also use a kernel for arrays of bits types of at least 1 MiB; other arrays are filled on the calling thread.

**Demonstration:**

On a system with two AMD EPYC 7502 (2 × 32 cores, 2 NUMA domains), the example (`N = 512_000_000`, workgroups of 1024), started with `-t T,0` for each thread count `T`, gets the following memory bandwidth, in GB/s:

| threads | initialization | `pinthreads(:numa)` | `pinthreads(:cores)` |
|--:|---|--:|--:|
| 64 | `init = :parallel` | 243 | 243 |
| 64 | `zeros` and `ones` | 241 | 242 |
| 32 | `init = :parallel` | 158 | |

A plain `Threads.@threads :static` loop with its threads pinned by ThreadPinning.jl gets 240 GB/s with 64 threads. Initializing serially puts all the memory in one domain. Without pinning, the threads that touch the memory first aren't reliably the ones that work on it later, and the numbers vary from run to run with where the operating system places the threads.

On a system with a single memory domain, pinning and the initialization don't change where memory is placed: on an AMD Ryzen 9 9950X with 8 or 16 threads, all configurations get 53–55 GB/s.
