module LineNumbers

using KernelAbstractions
using Test

const FILE = "kernel_source.jl"

collect_linenodes!(out, ::Any) = out
collect_linenodes!(out, node::LineNumberNode) = push!(out, node)
function collect_linenodes!(out, expr::Expr)
    for arg in expr.args
        collect_linenodes!(out, arg)
    end
    return out
end

"""
Expand `src` as if it had been written to `FILE` starting at line 1, and return
every `LineNumberNode` present in the expansion.
"""
function expanded_linenodes(src)
    toplevel = Meta.parseall(src; filename = FILE)
    nodes = LineNumberNode[]
    for arg in toplevel.args
        # `macroexpand` does not descend into `:toplevel`, so expand each
        # statement individually
        arg isa LineNumberNode && continue
        collect_linenodes!(nodes, macroexpand(@__MODULE__, arg))
    end
    return nodes
end

files(nodes) = unique(String.(getproperty.(nodes, :file)))
lines(nodes) = sort!(unique(getproperty.(nodes, :line)))

function linenumbers_testsuite()
    # A `@kernel` expansion must only refer back to the file it was written in.
    # Leaking `LineNumberNode`s that point into KernelAbstractions itself
    # misattributes the user's kernel for coverage and profiling tools.
    # https://github.com/JuliaGPU/KernelAbstractions.jl/issues/732
    @testset "simple kernel" begin
        nodes = expanded_linenodes(
            """
            @kernel function simple!(a, b)
                i = @index(Global)
                x = a[i]
                b[i] = x * 2
            end
            """
        )
        @test files(nodes) == [FILE]
        # line 1 covers the constructors, 2-4 the body
        @test lines(nodes) == [1, 2, 3, 4]
    end

    @testset "@synchronize and hoisted allocations" begin
        nodes = expanded_linenodes(
            """
            @kernel function sync!(a)
                i = @index(Local)
                lm = @localmem Float64 (8,)
                lm[i] = a[i]
                @synchronize
                a[i] = lm[i]
            end
            """
        )
        @test files(nodes) == [FILE]
        # `@localmem` is hoisted out of the workitem loop and the statement
        # after `@synchronize` starts a new one; both used to lose their line.
        @test lines(nodes) == [1, 2, 3, 4, 5, 6]
    end

    @testset "@uniform and @private" begin
        nodes = expanded_linenodes(
            """
            @kernel function alloc!(a)
                i = @index(Local)
                @uniform N = 8
                p = @private Float64 (1,)
                @private q = 0.0
                a[i] = N + p[1] + q
            end
            """
        )
        @test files(nodes) == [FILE]
        @test lines(nodes) == [1, 2, 3, 4, 5, 6]
    end

    @testset "kernel configuration" begin
        for config in ("inbounds=true", "unsafe_indices=true", "inbounds=true unsafe_indices=true")
            nodes = expanded_linenodes(
                """
                @kernel $config function configured!(a)
                    i = @index(Global)
                    a[i] = 2a[i]
                end
                """
            )
            @test files(nodes) == [FILE]
            @test lines(nodes) == [1, 2, 3]
        end
    end

    @testset "kernel-language macros" begin
        # These expand inside the kernel body, so they must not splice line
        # information of their own definition site into it either.
        nodes = expanded_linenodes(
            """
            @kernel function language!(a)
                i = @index(Global)
                gs = @groupsize()
                nd = @ndrange()
                @print("hello")
                a[i] = prod(gs) + prod(nd)
            end
            """
        )
        @test files(nodes) == [FILE]
        @test lines(nodes) == [1, 2, 3, 4, 5, 6]
    end

    return
end

end # module
