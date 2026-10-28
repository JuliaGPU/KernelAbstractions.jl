# Runs the benchmarks and writes their results, for the benchmark history at
# https://juliagpu.org/perf/:
#
#     julia --project=perf perf/runbenchmarks.jl [benchmarks.json]
#
# Benchmark names are `category/backend/mode/group/...`, with the categories of
# perf/benchmark-config.json. On the CPU backend, `serial` benchmarks run with one PoCL
# worker and `parallel` ones with a worker per physical core that this process may use (or
# `KA_BENCHMARK_THREADS`), with as many Julia threads, which the backend may run
# work-groups on. Every group of benchmarks runs in a fresh process, as the number of
# workers can't change once the backend has been used.
#
# Results that don't pass the checks below aren't written, so that they don't get published.

using BenchmarkTools, JSON, Statistics
# for the versions of the packages that the benchmarks use
using KernelAbstractions

const PERF = @__DIR__
const OUTPUT = length(ARGS) >= 1 ? ARGS[1] : "benchmarks.json"
const JULIA = `$(Base.julia_cmd()) --project=$(dirname(Base.active_project())) --startup-file=no`

# The number of physical cores, but no more than the CPUs this process may run on
function physical_cores()
    if Sys.islinux()
        cores = Set{Tuple{String, String}}()
        socket = ""
        for line in eachline("/proc/cpuinfo")
            parts = split(line, ':'; limit = 2)
            length(parts) == 2 || continue
            key, value = strip.(parts)
            key == "physical id" && (socket = value)
            key == "core id" && push!(cores, (socket, value))
        end
        available = parse(Int, readchomp(`nproc`))
        return isempty(cores) ? available : min(length(cores), available)
    elseif Sys.isapple()
        return parse(Int, readchomp(`sysctl -n hw.physicalcpu`))
    else
        return Sys.CPU_THREADS
    end
end

const PARALLEL_THREADS = parse(Int, get(ENV, "KA_BENCHMARK_THREADS", string(physical_cores())))

# The environment of a benchmark process: `threads` workers and Julia threads, and a PoCL
# kernel cache of its own, so that no kernel compiled by an earlier run is reused
function child_env(threads; cache = mktempdir())
    env = filter(((k, _),) -> !startswith(k, "POCL_") && k != "JULIA_KA_CPU_THREADS", ENV)
    env["JULIA_KA_CPU_THREADS"] = string(threads)
    env["JULIA_NUM_THREADS"] = "$threads,0"
    env["POCL_CACHE_DIR"] = cache
    env["OPENBLAS_NUM_THREADS"] = "1"
    return env
end

include("latency.jl")

# Runs the benchmarks of `group` (see suite.jl) in a fresh process
function run_suite(mode, group)
    file = tempname() * ".json"
    threads = mode == "serial" ? 1 : PARALLEL_THREADS
    run(setenv(`$JULIA $(joinpath(PERF, "suite.jl")) $mode $group $file`, child_env(threads)))
    results = Dict(join(keys, "/") => trial for (keys, trial) in BenchmarkTools.leaves(BenchmarkTools.load(file)[1]))
    rm(file)
    return results
end

# The launch benchmarks take microseconds, and vary more between processes than within
# one, so their samples are taken from several processes
const LAUNCH_PROCESSES = 3

function launch_benchmarks(mode)
    runs = [run_suite(mode, "launch") for _ in 1:LAUNCH_PROCESSES]
    results = Dict{String, BenchmarkTools.Trial}()
    for name in keys(first(runs))
        trials = [r[name] for r in runs]
        @info "$name, per process" minimum = [minimum(t.times) for t in trials] median = [median(t.times) for t in trials]
        # every process has to have taken all of its samples
        params = BenchmarkTools.Parameters(first(trials).params; samples = sum(t.params.samples for t in trials))
        results[name] = BenchmarkTools.Trial(
            params, reduce(vcat, [t.times for t in trials]),
            reduce(vcat, [t.gctimes for t in trials]),
            maximum(t.memory for t in trials), maximum(t.allocs for t in trials)
        )
    end
    return results
end

results = latency_benchmarks()
# launches are only measured with all workers (see launch.jl)
@info "Running the launch benchmarks"
merge!(results, launch_benchmarks("parallel"))
for mode in ("parallel", "serial")
    @info "Running the $mode benchmarks"
    merge!(results, run_suite(mode, "main"))
end

# Launches that don't allocate, which the history can't flag when they start to: it
# compares values by their ratio
const NON_ALLOCATING = [
    "latency/pocl/parallel/launch/dynamic-default", "latency/pocl/parallel/launch/partial-workgroup",
]
allocating = filter(name -> results[name].allocs != 0, NON_ALLOCATING)
isempty(allocating) || error("Launches that should not allocate did: $(join(allocating, ", "))")

# Benchmarks that ran out of time before taking all of their samples aren't comparable
# with earlier runs, so they are left out
for name in sort(collect(keys(results)))
    trial = results[name]
    if length(trial.times) < trial.params.samples
        @warn "Leaving out $name, which only took $(length(trial.times)) of $(trial.params.samples) samples"
        delete!(results, name)
    end
end

# Every statistic of every benchmark, and the environment they were measured in. Times
# are in nanoseconds.
function export_results(file, results)
    benchmarks = Dict{String, Any}()
    for (name, trial) in results
        times = trial.times
        benchmarks[name] = Dict(
            "min" => minimum(times), "p25" => quantile(times, 0.25),
            "median" => median(times), "p75" => quantile(times, 0.75),
            "mean" => mean(times), "max" => maximum(times),
            "std" => length(times) > 1 ? std(times) : nothing,
            "gctime" => median(trial.gctimes),
            "memory" => trial.memory, "allocs" => trial.allocs,
            "samples" => length(times), "evals" => trial.params.evals
        )
    end
    # the versions of every package loaded, to explain changes the commit itself does not
    packages = Dict(
        id.name => string(pkgversion(mod)) for (id, mod) in Base.loaded_modules
            if pkgversion(mod) !== nothing
    )
    env = Dict(
        "julia" => string(VERSION),
        "cpu" => Sys.cpu_info()[1].model,
        "os" => Sys.iswindows() ? "Windows" : readchomp(`uname -sr`),
        "threads" => PARALLEL_THREADS,
        "agent" => get(ENV, "BUILDKITE_AGENT_NAME", nothing),
        "package" => string(pkgversion(KernelAbstractions)),
        "packages" => packages
    )
    open(file, "w") do io
        JSON.print(io, Dict("version" => 2, "env" => env, "benchmarks" => benchmarks))
    end
    return
end
export_results(OUTPUT, results)
