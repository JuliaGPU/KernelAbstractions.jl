# Runs the benchmarks of one execution mode in this process, and saves their results:
#
#     julia --project=perf perf/suite.jl MODE GROUP OUTPUT
#
# with MODE `serial` or `parallel`, and GROUP `launch` (the launch-overhead benchmarks) or
# `main` (all others). `runbenchmarks.jl` starts this with as many Julia threads as the mode
# has workers, and `JULIA_KA_CPU_THREADS` set to that number: it has to be set before the
# backend is first used.

using KernelAbstractions
using BenchmarkTools
using Random
using Statistics

const MODE = ARGS[1]
const GROUP = ARGS[2]
const OUTPUT = ARGS[3]
MODE in ("serial", "parallel") || error("Unknown mode $MODE")
GROUP in ("launch", "main") || error("Unknown group $GROUP")

const BACKEND = CPU()

const SUITE = BenchmarkGroup()

# Add the benchmark that `f()` returns as `name`, if it belongs to this process: names are
# `category/backend/mode/group/...`. `f` also sets up and checks the benchmark, so that
# the other processes don't have to.
function bench(f, name)
    parts = split(name, "/")
    parts[3] == MODE || return
    (parts[4] == "launch") == (GROUP == "launch") || return
    group = SUITE
    for part in parts[1:(end - 1)]
        group = haskey(group, part) ? group[part] : (group[part] = BenchmarkGroup())
    end
    group[parts[end]] = f()
    return
end

# The samples of a benchmark: there's no tuning, so that the results of different runs are
# comparable, and one evaluation per sample, as the benchmarks run long enough (or are
# launches, which each have to be measured).
# `seconds` only guards against a benchmark that is much slower than expected; the
# runner reports benchmarks that didn't get all of their samples.
const KERNEL = (samples = 1000, seconds = 30)
const APPLICATION = (samples = 200, seconds = 30)
const LAUNCH = (samples = 10_000, seconds = 5)

# A benchmark of calling `f()`, after `setup()`
function measure(f, kind; setup = Returns(nothing))
    (; samples, seconds) = kind
    return @benchmarkable $f() setup = ($setup()) evals = 1 samples = samples seconds = seconds
end

if GROUP == "launch"
    include("launch.jl")
else
    include("kernels.jl")
    include("applications.jl")
end

results = run(SUITE; verbose = true)
BenchmarkTools.save(OUTPUT, results)
