# Measuring performance

Run under `nsight-cu`:

```sh
nv-nsight-cu-cli --nvtx --profile-from-start=off --section=SpeedOfLight julia --project=examples examples/performance.jl
```

## Results:

Collated results on a V100:

| Kernel          | Time   | Speed of Light Mem % |
| --------------- | ------ | -------------------- |
| naive (32, 32)  | 1.19ms | 65.06%               |
| naive (1024, 1) | 1.79ms | 56.13 %              |
| naive (1, 1024) | 3.03ms | 60.02 %              |

### Full output:
```
==PROF==   0: Naive transpose (32, 32)
    Section: GPU Speed Of Light
    ---------------------------------------------------------------------- --------------- ------------------------------
    Memory Frequency                                                         cycle/usecond                         878.88
    SOL FB                                                                               %                          38.16
    Elapsed Cycles                                                                   cycle                      1,447,874
    SM Frequency                                                             cycle/nsecond                           1.23
    Memory [%]                                                                           %                          65.93
    Duration                                                                       msecond                           1.17
    SOL L2                                                                               %                          19.08
    SOL TEX                                                                              %                          66.19
    SM Active Cycles                                                                 cycle                   1,440,706.40
    SM [%]                                                                               %                          23.56
    ---------------------------------------------------------------------- --------------- ------------------------------

  ptxcall___gpu_transpose_kernel_naive__430_2, 2020-Feb-20 22:42:24, Context 1, Stream 23

==PROF==   0: Naive transpose (1024, 1)
    Section: GPU Speed Of Light
    ---------------------------------------------------------------------- --------------- ------------------------------
    Memory Frequency                                                         cycle/usecond                         877.69
    SOL FB                                                                               %                          22.40
    Elapsed Cycles                                                                   cycle                      2,473,141
    SM Frequency                                                             cycle/nsecond                           1.23
    Memory [%]                                                                           %                          51.17
    Duration                                                                       msecond                           2.00
    SOL L2                                                                               %                          50.17
    SOL TEX                                                                              %                          51.27
    SM Active Cycles                                                                 cycle                   2,465,610.06
    SM [%]                                                                               %                          11.68
    ---------------------------------------------------------------------- --------------- ------------------------------

  ptxcall___gpu_transpose_kernel_naive__430_3, 2020-Feb-20 22:42:28, Context 1, Stream 25

==PROF==   0: Naive transpose (1, 1024)
    Section: GPU Speed Of Light
    ---------------------------------------------------------------------- --------------- ------------------------------
    Memory Frequency                                                         cycle/usecond                         876.69
    SOL FB                                                                               %                          17.88
    Elapsed Cycles                                                                   cycle                      3,737,127
    SM Frequency                                                             cycle/nsecond                           1.24
    Memory [%]                                                                           %                          60.02
    Duration                                                                       msecond                           3.02
    SOL L2                                                                               %                          60.02
    SOL TEX                                                                              %                          45.65
    SM Active Cycles                                                                 cycle                   3,732,591.59
    SM [%]                                                                               %                          12.56
    ---------------------------------------------------------------------- --------------- ------------------------------
```

## Code
````@eval
using Markdown
using KernelAbstractions
path = joinpath(dirname(pathof(KernelAbstractions)), "..", "examples/performance.jl")
Markdown.parse("""
```julia
$(read(path, String))
```
""")
````
