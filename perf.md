# Benchmark Report for */home/vchuravy/src/KernelAbstractions*

## Job Properties
* Time of benchmark: 6 Nov 2023 - 21:25
* Package commit: 609404
* Julia commit: 404750
* Julia command flags: None
* Environment variables: `KA_BACKEND => CPU` `JULIA_NUM_THREADS => auto`

## Results
Below is a table of this job's results, obtained by running the benchmarks.
The values listed in the `ID` column have the structure `[parent_group, child_group, ..., key]`, and can be used to
index into the BaseBenchmarks suite to retrieve the corresponding benchmarks.
The percentages accompanying time and memory values in the below table are noise tolerances. The "true"
time/memory value for a given benchmark is expected to fall within this percentage of the reported value.
An empty cell means that the value was zero.

| ID                                                                | time            | GC time | memory         | allocations |
|-------------------------------------------------------------------|----------------:|--------:|---------------:|------------:|
| `["saxpy", "default", "Float16", "N=1024, workgroup=(1024,)"]`    |   1.664 μs (5%) |         | 272 bytes (1%) |          11 |
| `["saxpy", "default", "Float16", "N=1048576, workgroup=(1024,)"]` | 185.800 μs (5%) |         | 11.66 KiB (1%) |         161 |
| `["saxpy", "default", "Float16", "N=16384, workgroup=(1024,)"]`   |  13.630 μs (5%) |         | 11.66 KiB (1%) |         161 |
| `["saxpy", "default", "Float16", "N=2048, workgroup=(1024,)"]`    |   5.575 μs (5%) |         |  1.92 KiB (1%) |          34 |
| `["saxpy", "default", "Float16", "N=256, workgroup=(1024,)"]`     |   1.287 μs (5%) |         | 272 bytes (1%) |          11 |
| `["saxpy", "default", "Float16", "N=262144, workgroup=(1024,)"]`  |  56.480 μs (5%) |         | 11.66 KiB (1%) |         161 |
| `["saxpy", "default", "Float16", "N=32768, workgroup=(1024,)"]`   |  15.390 μs (5%) |         | 11.66 KiB (1%) |         161 |
| `["saxpy", "default", "Float16", "N=4096, workgroup=(1024,)"]`    |   6.357 μs (5%) |         |  3.27 KiB (1%) |          52 |
| `["saxpy", "default", "Float16", "N=512, workgroup=(1024,)"]`     |   1.562 μs (5%) |         | 272 bytes (1%) |          11 |
| `["saxpy", "default", "Float16", "N=64, workgroup=(1024,)"]`      |   1.079 μs (5%) |         | 272 bytes (1%) |          11 |
| `["saxpy", "default", "Float16", "N=65536, workgroup=(1024,)"]`   |  23.800 μs (5%) |         | 11.66 KiB (1%) |         161 |
| `["saxpy", "default", "Float32", "N=1024, workgroup=(1024,)"]`    | 859.643 ns (5%) |         | 272 bytes (1%) |          11 |
| `["saxpy", "default", "Float32", "N=1048576, workgroup=(1024,)"]` | 299.990 μs (5%) |         | 11.66 KiB (1%) |         161 |
| `["saxpy", "default", "Float32", "N=16384, workgroup=(1024,)"]`   |  11.660 μs (5%) |         | 11.66 KiB (1%) |         161 |
| `["saxpy", "default", "Float32", "N=2048, workgroup=(1024,)"]`    |   4.066 μs (5%) |         |  1.92 KiB (1%) |          34 |
| `["saxpy", "default", "Float32", "N=256, workgroup=(1024,)"]`     |   1.004 μs (5%) |         | 272 bytes (1%) |          11 |
| `["saxpy", "default", "Float32", "N=262144, workgroup=(1024,)"]`  |  90.031 μs (5%) |         | 11.66 KiB (1%) |         161 |
| `["saxpy", "default", "Float32", "N=32768, workgroup=(1024,)"]`   |  13.660 μs (5%) |         | 11.66 KiB (1%) |         161 |
| `["saxpy", "default", "Float32", "N=4096, workgroup=(1024,)"]`    |   5.296 μs (5%) |         |  3.27 KiB (1%) |          52 |
| `["saxpy", "default", "Float32", "N=512, workgroup=(1024,)"]`     |   1.017 μs (5%) |         | 272 bytes (1%) |          11 |
| `["saxpy", "default", "Float32", "N=64, workgroup=(1024,)"]`      |   1.005 μs (5%) |         | 272 bytes (1%) |          11 |
| `["saxpy", "default", "Float32", "N=65536, workgroup=(1024,)"]`   |  20.450 μs (5%) |         | 11.66 KiB (1%) |         161 |
| `["saxpy", "default", "Float64", "N=1024, workgroup=(1024,)"]`    | 912.222 ns (5%) |         | 272 bytes (1%) |          11 |
| `["saxpy", "default", "Float64", "N=1048576, workgroup=(1024,)"]` | 548.010 μs (5%) |         | 11.66 KiB (1%) |         161 |
| `["saxpy", "default", "Float64", "N=16384, workgroup=(1024,)"]`   |  11.710 μs (5%) |         | 11.66 KiB (1%) |         161 |
| `["saxpy", "default", "Float64", "N=2048, workgroup=(1024,)"]`    |   4.180 μs (5%) |         |  1.92 KiB (1%) |          34 |
| `["saxpy", "default", "Float64", "N=256, workgroup=(1024,)"]`     |   1.038 μs (5%) |         | 272 bytes (1%) |          11 |
| `["saxpy", "default", "Float64", "N=262144, workgroup=(1024,)"]`  | 150.330 μs (5%) |         | 11.66 KiB (1%) |         161 |
| `["saxpy", "default", "Float64", "N=32768, workgroup=(1024,)"]`   |  14.590 μs (5%) |         | 11.66 KiB (1%) |         161 |
| `["saxpy", "default", "Float64", "N=4096, workgroup=(1024,)"]`    |   5.232 μs (5%) |         |  3.27 KiB (1%) |          52 |
| `["saxpy", "default", "Float64", "N=512, workgroup=(1024,)"]`     |   1.034 μs (5%) |         | 272 bytes (1%) |          11 |
| `["saxpy", "default", "Float64", "N=64, workgroup=(1024,)"]`      |   1.037 μs (5%) |         | 272 bytes (1%) |          11 |
| `["saxpy", "default", "Float64", "N=65536, workgroup=(1024,)"]`   |  27.440 μs (5%) |         | 11.66 KiB (1%) |         161 |
| `["saxpy", "static", "Float16", "N=1024, workgroup=(1024,)"]`     |   1.848 μs (5%) |         | 448 bytes (1%) |          19 |
| `["saxpy", "static", "Float16", "N=1048576, workgroup=(1024,)"]`  | 138.041 μs (5%) |         | 11.59 KiB (1%) |         169 |
| `["saxpy", "static", "Float16", "N=16384, workgroup=(1024,)"]`    |  11.589 μs (5%) |         | 11.59 KiB (1%) |         169 |
| `["saxpy", "static", "Float16", "N=2048, workgroup=(1024,)"]`     |   4.894 μs (5%) |         |  2.08 KiB (1%) |          42 |
| `["saxpy", "static", "Float16", "N=256, workgroup=(1024,)"]`      |   2.397 μs (5%) |         | 448 bytes (1%) |          19 |
| `["saxpy", "static", "Float16", "N=262144, workgroup=(1024,)"]`   |  30.000 μs (5%) |         | 11.59 KiB (1%) |         169 |
| `["saxpy", "static", "Float16", "N=32768, workgroup=(1024,)"]`    |  13.160 μs (5%) |         | 11.59 KiB (1%) |         169 |
| `["saxpy", "static", "Float16", "N=4096, workgroup=(1024,)"]`     |   6.305 μs (5%) |         |  3.39 KiB (1%) |          60 |
| `["saxpy", "static", "Float16", "N=512, workgroup=(1024,)"]`      |   2.727 μs (5%) |         | 448 bytes (1%) |          19 |
| `["saxpy", "static", "Float16", "N=64, workgroup=(1024,)"]`       |   2.152 μs (5%) |         | 448 bytes (1%) |          19 |
| `["saxpy", "static", "Float16", "N=65536, workgroup=(1024,)"]`    |  15.500 μs (5%) |         | 11.59 KiB (1%) |         169 |
| `["saxpy", "static", "Float32", "N=1024, workgroup=(1024,)"]`     |   1.838 μs (5%) |         | 448 bytes (1%) |          19 |
| `["saxpy", "static", "Float32", "N=1048576, workgroup=(1024,)"]`  | 320.820 μs (5%) |         | 11.59 KiB (1%) |         169 |
| `["saxpy", "static", "Float32", "N=16384, workgroup=(1024,)"]`    |  13.130 μs (5%) |         | 11.59 KiB (1%) |         169 |
| `["saxpy", "static", "Float32", "N=2048, workgroup=(1024,)"]`     |   4.822 μs (5%) |         |  2.08 KiB (1%) |          42 |
| `["saxpy", "static", "Float32", "N=256, workgroup=(1024,)"]`      |   2.518 μs (5%) |         | 448 bytes (1%) |          19 |
| `["saxpy", "static", "Float32", "N=262144, workgroup=(1024,)"]`   |  62.500 μs (5%) |         | 11.59 KiB (1%) |         169 |
| `["saxpy", "static", "Float32", "N=32768, workgroup=(1024,)"]`    |  13.780 μs (5%) |         | 11.59 KiB (1%) |         169 |
| `["saxpy", "static", "Float32", "N=4096, workgroup=(1024,)"]`     |   6.137 μs (5%) |         |  3.39 KiB (1%) |          60 |
| `["saxpy", "static", "Float32", "N=512, workgroup=(1024,)"]`      |   2.518 μs (5%) |         | 448 bytes (1%) |          19 |
| `["saxpy", "static", "Float32", "N=64, workgroup=(1024,)"]`       |   2.522 μs (5%) |         | 448 bytes (1%) |          19 |
| `["saxpy", "static", "Float32", "N=65536, workgroup=(1024,)"]`    |  17.460 μs (5%) |         | 11.59 KiB (1%) |         169 |
| `["saxpy", "static", "Float64", "N=1024, workgroup=(1024,)"]`     |   1.990 μs (5%) |         | 448 bytes (1%) |          19 |
| `["saxpy", "static", "Float64", "N=1048576, workgroup=(1024,)"]`  | 591.200 μs (5%) |         | 11.59 KiB (1%) |         169 |
| `["saxpy", "static", "Float64", "N=16384, workgroup=(1024,)"]`    |  13.830 μs (5%) |         | 11.59 KiB (1%) |         169 |
| `["saxpy", "static", "Float64", "N=2048, workgroup=(1024,)"]`     |   5.042 μs (5%) |         |  2.08 KiB (1%) |          42 |
| `["saxpy", "static", "Float64", "N=256, workgroup=(1024,)"]`      |   2.481 μs (5%) |         | 448 bytes (1%) |          19 |
| `["saxpy", "static", "Float64", "N=262144, workgroup=(1024,)"]`   | 152.971 μs (5%) |         | 11.59 KiB (1%) |         169 |
| `["saxpy", "static", "Float64", "N=32768, workgroup=(1024,)"]`    |  16.120 μs (5%) |         | 11.59 KiB (1%) |         169 |
| `["saxpy", "static", "Float64", "N=4096, workgroup=(1024,)"]`     |   5.740 μs (5%) |         |  3.39 KiB (1%) |          60 |
| `["saxpy", "static", "Float64", "N=512, workgroup=(1024,)"]`      |   2.494 μs (5%) |         | 448 bytes (1%) |          19 |
| `["saxpy", "static", "Float64", "N=64, workgroup=(1024,)"]`       |   2.453 μs (5%) |         | 448 bytes (1%) |          19 |
| `["saxpy", "static", "Float64", "N=65536, workgroup=(1024,)"]`    |  25.430 μs (5%) |         | 11.59 KiB (1%) |         169 |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["saxpy", "default", "Float16"]`
- `["saxpy", "default", "Float32"]`
- `["saxpy", "default", "Float64"]`
- `["saxpy", "static", "Float16"]`
- `["saxpy", "static", "Float32"]`
- `["saxpy", "static", "Float64"]`

## Julia versioninfo
```
Julia Version 1.10.0-beta3
Commit 404750f8586 (2023-10-03 12:53 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      "Arch Linux"
  uname: Linux 6.5.5-arch1-1 #1 SMP PREEMPT_DYNAMIC Sat, 23 Sep 2023 22:55:13 +0000 x86_64 unknown
  CPU: AMD Ryzen 7 3700X 8-Core Processor: 
                 speed         user         nice          sys         idle          irq
       #1-16  4049 MHz     509487 s       3939 s     154394 s    2536060 s      26100 s
  Memory: 125.69866943359375 GB (91917.546875 MB free)
  Uptime: 542756.6 sec
  Load Avg:  11.36  9.59  5.14
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)
  Threads: 23 on 16 virtual cores
```