# Profiling

KernelAbstractions can put named ranges on the timeline of a tracing profiler, such as
NVIDIA Nsight Systems or Intel VTune, so that you can see which part of your program a
stretch of kernels belongs to. Annotations are cheap when no profiler is listening: a
single atomic load, and the label isn't even built.

## Annotating code

Wrap code in [`@profiling_range`](@ref):

```julia
@profiling_range "volume integral" begin
    volume_integral!(du, u, backend)
end
```

Ranges can be grouped with a `domain`, which maps to an NVTX or ITT domain:

```julia
@profiling_range "time step $i" domain = "Trixi" begin
    step!(integrator)
end
```

Kernel launches are annotated with the kernel's name automatically. For an instantaneous
event, use [`profiling_mark`](@ref), and for ranges that don't follow the structure of the
code, [`profiling_range_start`](@ref KernelAbstractions.profiling_range_start) and
[`profiling_range_end`](@ref KernelAbstractions.profiling_range_end).

## Built-in profiler

To see where time goes without an external profiler, run code under
[`KernelAbstractions.@profile`](@ref KernelAbstractions.@profile). It records the ranges
and kernel launches of an expression, and summarizes them:

```julia-repl
julia> KernelAbstractions.@profile for i in 1:10
           @profiling_range "step" begin
               mul2(backend)(A; ndrange = length(A))
               add(backend)(A, B; ndrange = length(A))
           end
       end
Profiled 6.04 ms, recording 30 ranges.

 Time (%)  Total time  Calls  Avg time  Min time  Max time  Name
 ────────  ──────────  ─────  ────────  ────────  ────────  ────
   92.6 %     5.59 ms     10    559 µs    298 µs    2.9 ms  step
   57.6 %     3.48 ms     10    348 µs    109 µs   2.48 ms  mul2
   37.6 %     2.27 ms     10    227 µs    179 µs    400 µs  add
```

Kernel launches synchronize their backend while profiling, so that their ranges measure the
kernel rather than its launch; pass `synchronize = false` to measure launches. Pass
`trace = true` to list every range in order instead. The first call of a kernel includes
its compilation, so profile a warmed-up run.

`@profile` records the task running the expression and the tasks it spawns, e.g. with
[`KernelAbstractions.@spawn`](@ref KernelAbstractions.@spawn), but not other tasks, so
profiles can run concurrently. Wait for spawned tasks within the expression, e.g. with
`@sync`, as what they record after it returns is lost.

## Profilers

Ranges are recorded on the host threads of the process, which is how NVTX, ITT and
roctx work: it is the profiler that attributes the device work launched within a range to
it. So which profiler records the ranges depends on what the process runs under, not on the
backend: running the CPU backend under Nsight Systems gives NVTX ranges, and a GPU backend
under VTune gives ITT tasks.

Ranges go to every registered [`Tracer`](@ref KernelAbstractions.Tracer). These come with
KernelAbstractions, and only register themselves when their profiler is attached:

- **Nsight Systems**: load [NVTX.jl](https://github.com/JuliaGPU/NVTX.jl) (CUDA.jl loads it
  too) and run under `nsys profile --trace=nvtx,...`.
- **rocprof**: load [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) and run under
  `rocprofv3 --marker-trace` (or the legacy `rocprof --roctx-trace`). Ranges are recorded
  with roctx, which has no domains, so a `domain` other than `"KernelAbstractions"` prefixes
  the label.
- **Intel VTune**: load [IntelITT.jl](https://github.com/JuliaPerf/IntelITT.jl) and run
  under VTune.
- **NVTXT**, a text format that Nsight Systems imports, to trace without a profiler: set
  `JULIA_KA_NVTXT=1` to write `ka-<pid>.nvtxt` to the working directory, or set it to a
  path, in which `%p` is replaced by the process id. Then
  ```sh
  ImportNvtxt --cmd create --nvtxt ka-1234.nvtxt -o report.nsys-rep
  ```
  To trace only part of a program, register an [`NVTXTTracer`](@ref
  KernelAbstractions.NVTXTTracer) yourself:
  ```julia
  tracer = KernelAbstractions.register_tracer!(KernelAbstractions.NVTXTTracer("trace.nvtxt"))
  run_simulation()
  KernelAbstractions.unregister_tracer!(tracer)
  close(tracer)
  ```

Other profilers are supported by subtyping
[`Tracer`](@ref KernelAbstractions.Tracer).

## API

```@docs
@profiling_range
profiling_mark
KernelAbstractions.@profile
KernelAbstractions.ProfileResults
KernelAbstractions.profiling_active
KernelAbstractions.profiling_range_start
KernelAbstractions.profiling_range_end
KernelAbstractions.Tracer
KernelAbstractions.register_tracer!
KernelAbstractions.unregister_tracer!
KernelAbstractions.NVTXTTracer
```
