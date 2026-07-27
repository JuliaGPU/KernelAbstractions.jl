module Coverage

using KernelAbstractions
using Test

# Script exercising the constructs whose line information used to be dropped or
# misattributed by `@kernel`. Every line of both kernel bodies must show up as
# tracked in the resulting coverage data.
const SCRIPT = """
using KernelAbstractions

@kernel function mul2!(a)
    i = @index(Global)
    x = a[i]
    a[i] = 2 * x
end

@kernel function sync2!(a)
    i = @index(Local)
    lm = @localmem Float64 (8,)
    lm[i] = a[i]
    @synchronize
    a[i] = lm[i] + 1
end

function main()
    a = ones(8)
    mul2!(CPU(), 8)(a, ndrange = 8)
    sync2!(CPU(), 8)(a, ndrange = 8)
    synchronize(CPU())
    a == fill(3.0, 8) || error("unexpected result: \$a")
    return
end

main()
"""

const SCRIPT_NAME = "kernels.jl"

"""
Line numbers of `SCRIPT_NAME` that the LCOV tracefile reports as tracked.

Records look like `SF:<path>`, followed by one `DA:<line>,<count>` per tracked
line, terminated by `end_of_record`.
"""
function tracked_lines(tracefile)
    lines = Set{Int}()
    in_script = false
    for line in eachline(tracefile)
        if startswith(line, "SF:")
            in_script = basename(line[4:end]) == SCRIPT_NAME
        elseif line == "end_of_record"
            in_script = false
        elseif in_script && startswith(line, "DA:")
            push!(lines, parse(Int, first(split(line[4:end], ','))))
        end
    end
    return lines
end

function run_covered(dir)
    script = joinpath(dir, SCRIPT_NAME)
    write(script, SCRIPT)
    log = joinpath(dir, "log.txt")
    # Write to an LCOV tracefile rather than using `--code-coverage=user`: the
    # latter drops a `.cov` file next to every user source file it tracks, which
    # would litter both the checkout and the depot, and would perturb the outer
    # coverage report when the suite itself runs under `Pkg.test(coverage=true)`.
    tracefile = joinpath(dir, "lcov.info")

    cmd = `$(Base.julia_cmd()) --startup-file=no --code-coverage=$tracefile
        --project=$(Base.active_project()) $script`
    proc = run(pipeline(ignorestatus(cmd); stdout = log, stderr = log))
    if !success(proc)
        @error "coverage subprocess failed" output = read(log, String)
    end
    @test success(proc)
    @test isfile(tracefile)
    isfile(tracefile) || return nothing

    return tracked_lines(tracefile)
end

function coverage_testsuite()
    # GPUCompiler records device coverage by visiting the source location of
    # every `:code_coverage_effect` while compiling, so a kernel whose line
    # information points into KernelAbstractions reads as untracked in the
    # user's file. https://github.com/JuliaGPU/KernelAbstractions.jl/issues/732
    mktempdir() do dir
        tracked = run_covered(dir)
        tracked === nothing && return

        srclines = split(SCRIPT, '\n')
        @testset "$needle" for needle in (
                "@kernel function mul2!",
                "i = @index(Global)",
                "x = a[i]",
                "a[i] = 2 * x",
                "@kernel function sync2!",
                "i = @index(Local)",
                "lm = @localmem Float64 (8,)",
                "lm[i] = a[i]",
                "@synchronize",
                "a[i] = lm[i] + 1",
            )
            line = findfirst(src -> occursin(needle, src), srclines)
            @test line !== nothing
            @test line in tracked
        end
    end
    return
end

end # module
