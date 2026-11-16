# Latency benchmarks, each sample of which is a fresh Julia process, run before anything
# else. The processes time what they measure themselves (see `latency/`), except for
# `startup`. Every process gets an empty PoCL kernel cache of its own, except for the
# `warm-cache` samples, which share one that a previous process populated.

# The time that `script` reports, in `samples` fresh processes
function process_trial(script, args...; samples, threads = 1, cache = nothing)
    times = map(1:samples) do _
        dir = something(cache, mktempdir())
        cmd = `$JULIA $(joinpath(PERF, "latency", script)) $args`
        out = readchomp(setenv(cmd, child_env(threads; cache = dir)))
        cache === nothing && rm(dir; recursive = true)
        parse(Float64, last(split(out, '\n'))) * 1.0e9
    end
    return trial(times)
end

function trial(times)
    params = BenchmarkTools.Parameters(; evals = 1, samples = length(times))
    return BenchmarkTools.Trial(params, times, zeros(length(times)), 0, 0)
end

function latency_benchmarks()
    results = Dict{String, BenchmarkTools.Trial}()

    @info "Measuring the start-up time of Julia"
    startup = `$JULIA -e 0`
    results["latency/host/startup"] =
        trial([@elapsed(run(setenv(startup, child_env(1)))) * 1.0e9 for _ in 1:20])

    @info "Measuring the time to precompile KernelAbstractions"
    results["latency/host/precompile/ka"] = process_trial("precompile.jl"; samples = 3)

    @info "Measuring the time to load KernelAbstractions"
    results["latency/host/import/ka"] = process_trial("import.jl"; samples = 20)

    @info "Measuring the time to the first kernel"
    results["latency/pocl/serial/first-kernel/cold-cache"] =
        process_trial("first_kernel.jl"; samples = 10)
    mktempdir() do cache
        process_trial("first_kernel.jl"; samples = 1, cache)
        results["latency/pocl/serial/first-kernel/warm-cache"] =
            process_trial("first_kernel.jl"; samples = 10, cache)
    end

    @info "Measuring the time to compile kernels"
    for case in ("new-kernel", "localmem-barrier", "groupreduce", "new-workgroupsize")
        results["latency/pocl/serial/compile/$case"] =
            process_trial("compile.jl", case; samples = 10)
    end

    return results
end
