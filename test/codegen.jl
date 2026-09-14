module Codegen

using Test

const SCRIPT = joinpath(@__DIR__, "codegen_checks.jl")

"""
Run codegen_checks.jl in a subprocess and return whether it succeeded.

The checks assert properties of optimized device code, which the flags `Pkg.test` sets
distort, so neither is inherited:

  * `--check-bounds=yes`, the `Pkg.test` default, forces bounds checks on regardless of
    `@inbounds`. Several checks assert that `@inbounds` removes the out-of-bounds path,
    which is not merely untestable but false under it.
  * `--code-coverage` instruments the kernels being inspected, and under
    `Pkg.test(coverage=true)` would also fold this subprocess into the outer report.
"""
function run_checks(log)
    inherited = arg -> startswith(arg, "--check-bounds") || startswith(arg, "--code-coverage")
    julia = Cmd(filter(!inherited, Base.julia_cmd().exec))
    cmd = `$julia --startup-file=no --check-bounds=auto
        --project=$(Base.active_project()) $SCRIPT`
    proc = run(pipeline(ignorestatus(cmd); stdout = log, stderr = log))
    if !success(proc)
        @error "codegen subprocess failed" output = read(log, String)
    end
    return success(proc)
end

function codegen_testsuite()
    mktempdir() do dir
        @test run_checks(joinpath(dir, "log.txt"))
    end
    return
end

end # module
