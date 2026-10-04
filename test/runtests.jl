using KernelAbstractions
using Random
using Test

include("quality_assurance.jl")
include("linenumbers.jl")
include("coverage.jl")
include("testsuite.jl")
include("codegen.jl")

@testset "Quality assurance" begin
    quality_assurance_testsuite()
end

@testset "Line numbers" begin
    LineNumbers.linenumbers_testsuite()
end

@testset "Coverage" begin
    Coverage.coverage_testsuite()
end

@testset "Precompilation" begin
    # without it, the first kernel launch on the CPU back-end takes many seconds
    workloads = Base.JLOptions().use_compiled_modules == 1 &&
        KernelAbstractions.PrecompileTools.workload_enabled(KernelAbstractions) &&
        KernelAbstractions.launch_in_workload
    @test KernelAbstractions.precompiled_launch[] skip = !workloads
end

KernelAbstractions.versioninfo(POCLBackend())
@info "Configuration" pocl = KernelAbstractions.POCL.nanoOpenCL.pocl_standalone_jll.libpocl

import KernelAbstractions.POCL: POCL, @opencl, @device_code_llvm

module KernelInterfaceTests
    import KernelInterface
    using Test
    include(joinpath(pkgdir(KernelInterface), "test", "testsuite.jl"))
end
@testset "POCL KernelInterface" begin
    KernelInterfaceTests.Testsuite.testsuite(POCLBackend(), Array)
end

@testset "POCL thread count" begin
    # in fresh processes, as PoCL only reads its configuration once
    julia = Cmd(filter(arg -> !startswith(arg, "--code-coverage"), Base.julia_cmd().exec))
    script = "using KernelAbstractions: POCL; print(POCL.device().max_compute_units)"
    vars = ("JULIA_KA_CPU_THREADS", POCL.cl.pocl_thread_variables...)
    function compute_units(env...; threads = 2)
        cmd = `$julia --startup-file=no --threads=$threads
            --project=$(Base.active_project()) -e $script`
        cmd_env = filter(kv -> !(first(kv) in vars), copy(ENV))
        return parse(Int, readchomp(setenv(cmd, cmd_env..., env...)))
    end

    # as many workers as Julia has threads
    @test compute_units(threads = 3) == 3
    # unless PoCL is configured otherwise
    @test compute_units("POCL_CPU_MAX_CU_COUNT" => "5") == 5
    # but KernelAbstractions' variable takes precedence
    @test compute_units("JULIA_KA_CPU_THREADS" => "4") == 4
    @test compute_units("JULIA_KA_CPU_THREADS" => "4", "POCL_MAX_PTHREAD_COUNT" => "5") == 4
end

@testset "POCL float atomics" begin
    # pocl's CPU device natively supports float add and min/max atomics in both global
    # and local memory, so the SPIR-V extensions guarding them must be permitted
    dev = POCL.device()
    exts = split(POCL.default_spirv_extensions(dev), ",")
    @test "+SPV_EXT_shader_atomic_float_add" in exts
    @test "+SPV_EXT_shader_atomic_float_min_max" in exts
    @test dev.half_fp_atomic_capabilities == 0
    # an explicit list overrides the device-derived default
    config = POCL.compiler_config(dev; extensions = "+SPV_KHR_expect_assume")
    @test config.target.extensions == "+SPV_KHR_expect_assume"
end

# `randn`/`randexp` for Float16 route through Random's table-free fallback, whose polar
# transform overflows in Float16 and whose `log1p` isn't available for Float16 on the
# device. The device overlays compute in Float32 and convert, so results stay finite.
if "cl_khr_fp16" in POCL.device().extensions
    @testset "POCL device RNG: Float16" begin
        @kernel function f16_rng_kernel(A, B)
            i = @index(Global, Linear)
            @inbounds A[i] = Random.randn(Float16)
            @inbounds B[i] = Random.randexp(Float16)
        end

        # the overflow this guards against hits a few hundred values in 2^20 draws,
        # so a small sample would not catch a regression
        len = 2^20
        a = KernelAbstractions.zeros(POCLBackend(), Float16, len)
        b = KernelAbstractions.zeros(POCLBackend(), Float16, len)
        f16_rng_kernel(POCLBackend())(a, b; ndrange = len)
        KernelAbstractions.synchronize(POCLBackend())
        @test all(isfinite, a)
        @test all(isfinite, b)
    end
end

# Julia doesn't turn a splat of more than 32 elements into a direct call, so a launch with
# many arguments allocates unless every layer passes them on as a tuple
@testset "POCL launch with many arguments" begin
    xs = [Symbol(:x, i) for i in 1:40]
    mod = @eval module $(gensym())
    using KernelAbstractions
    @kernel function few!(A, x1, x2, x3, x4)
        I = @index(Global, Linear)
        @inbounds A[I] = x1 + x2 + x3 + x4
    end
    @kernel function many!(A, $(xs...))
        I = @index(Global, Linear)
        @inbounds A[I] = $(foldl((a, b) -> :($a + $b), xs))
    end
    # the arguments are written out, since splatting them here would allocate too
    launch_few(k, A) = k(A, $((1:4)...); ndrange = length(A))
    launch_many(k, A) = k(A, $((1:40)...); ndrange = length(A))
    end

    A = zeros(Int, 16)
    few = mod.few!(CPU(), 16)
    many = mod.many!(CPU(), 16)
    mod.launch_few(few, A)
    mod.launch_many(many, A)
    @test all(==(sum(1:40)), A)
    # waiting for a kernel allocates when it involves a completion callback, which depends
    # on how long the kernel takes, so measure launches that wait by blocking instead
    allocated(launch, k, A) = @allocated launch(k, A)
    POCL.cl.blocking_waits[] = true
    try
        allocated(mod.launch_many, many, A)
        @test allocated(mod.launch_many, many, A) <= allocated(mod.launch_few, few, A)
    finally
        POCL.cl.blocking_waits[] = false
    end
end

@testset "POCL exceptions" begin
    @kernel function exception_kernel!(A)
        I = @index(Global, Linear)
        A[I + 1] = 1
    end
    exception_opencl!(A) = (A[POCL.get_global_id() + 1] = 1; return)
    # the device prints details about the exception
    quietly(f) = redirect_stdout(f, devnull)

    A = zeros(Int, 4)
    @test_throws POCL.KernelException quietly(() -> exception_kernel!(CPU())(A; ndrange = 4))
    @test A == [0, 1, 1, 1]
    @test_throws POCL.KernelException quietly(() -> @opencl global_size = 4 exception_opencl!(A))

    # launches after it are fine
    exception_kernel!(CPU())(A; ndrange = 3)

    # and launches from other tasks don't see it
    function launch(ndrange)
        try
            exception_kernel!(CPU())(zeros(Int, 4); ndrange)
            return false
        catch err
            err isa POCL.KernelException || rethrow()
            return true
        end
    end
    failing, succeeding = quietly() do
        tasks = (@async([launch(4) for _ in 1:10]), @async([launch(3) for _ in 1:10]))
        fetch.(tasks)
    end
    @test all(failing)
    @test !any(succeeding)
end

@testset "POCL exception output" begin
    exception_kernel!(A) = (A[POCL.get_global_id() + 4] = 1; return)
    function output(debug_level)
        return mktemp() do path, io
            redirect_stdout(io) do
                # in many work-groups, which all throw
                @test_throws POCL.KernelException @opencl global_size = 64 local_size = 1 debug_level exception_kernel!(zeros(Int, 4))
                Libc.flush_cstdio()
            end
            close(io)
            read(path, String)
        end
    end

    @test isempty(output(0))
    # of all work-items that throw, only one reports the exception
    out = output(1)
    @test count("ERROR: ", out) == 1
    @test occursin("BoundsError", out)
    out = output(2)
    @test count("ERROR: ", out) == 1
    @test occursin("Stacktrace:", out)
    @test occursin("throw_boundserror", out)
end

@testset "POCL compilation cache" begin
    mod = @eval module $(gensym())
    @noinline child() = return
    kernel() = child()
    end

    count() = POCL.compilations[]
    launch() = @opencl mod.kernel()

    # the initial launch compiles
    n = count()
    Base.invokelatest(launch)
    @test count() == n + 1

    # a second launch hits the cache
    Base.invokelatest(launch)
    @test count() == n + 1

    # jobs differing only in codegen-level settings get their own artifacts...
    POCL.clfunction(mod.kernel, Tuple{}; name = "custom")
    @test count() == n + 2
    # ... which are cached as well
    POCL.clfunction(mod.kernel, Tuple{}; name = "custom")
    @test count() == n + 2

    # reflection observes already-compiled kernels (by forcing recompilation,
    # which must leave the cached entry in a usable state)
    @test !isempty(sprint(io -> (@device_code_llvm io = io Base.invokelatest(launch))))
    n = count()
    Base.invokelatest(launch)
    @test count() == n

    # redefining the kernel recompiles...
    @eval mod kernel() = (child(); child())
    Base.invokelatest(launch)
    @test count() == n + 1
    # ... as does redefining a callee
    @eval mod @noinline child() = nothing
    Base.invokelatest(launch)
    @test count() == n + 2
end

@testset "POCL session" begin
    # in a fresh process, to check initialization, and with threads, which the tests
    # usually don't have
    julia = Cmd(filter(arg -> !startswith(arg, "--code-coverage"), Base.julia_cmd().exec))
    cmd = `$julia --startup-file=no --threads=4 --project=$(Base.active_project())
        $(joinpath(@__DIR__, "pocl_session.jl"))`
    @test success(pipeline(cmd; stdout, stderr))
end

@testset "CPU Codegen" begin
    Codegen.codegen_testsuite()
end

@testset "Device code reflection" begin
    @kernel function reflect_mul2(A)
        i = @index(Global, Linear)
        @inbounds A[i] = 2 * A[i]
    end

    A = KernelAbstractions.ones(POCLBackend(), Float32, 64)
    ir = sprint() do io
        KernelAbstractions.@device_code_llvm io = io debuginfo = :none reflect_mul2(POCLBackend(), 16)(A, ndrange = 64)
    end
    @test occursin("reflect_mul2", ir)
    # the wrapped expression is evaluated, not just compiled
    KernelAbstractions.synchronize(POCLBackend())
    @test all(==(2.0f0), A)
end

@kernel function busy_kernel!(A, n)
    I = @index(Global)
    acc = 0.0f0
    for j in 1:n
        acc += sin(Float32(j) + acc)
    end
    @inbounds A[I] = acc
end

# POCL launches wait for the kernel to finish, but let other tasks run in the meantime
@testset "POCL cooperative launches" begin
    A = zeros(Float32, 1024)
    kernel = busy_kernel!(POCLBackend())
    kernel(A, 1; ndrange = length(A))   # compile
    n = 1000
    while @elapsed(kernel(A, n; ndrange = length(A))) < 0.1
        n *= 2
    end

    ran = Ref(false)
    task = @async ran[] = true
    kernel(A, n; ndrange = length(A))
    @test ran[]
    wait(task)
end

# not part of the shared testsuite: not every back-end supports bits-union arrays
@testset "POCL zeros/ones of bits-union types" begin
    for T in (Union{Missing, Bool}, Union{Missing, Int32})
        Z = KernelAbstractions.zeros(POCLBackend(), T, 3)
        @test eltype(Z) == T
        @test all(x -> !ismissing(x) && iszero(x), Z)

        O = KernelAbstractions.ones(POCLBackend(), T, 3)
        @test eltype(O) == T
        @test all(x -> !ismissing(x) && isone(x), O)
    end
end

# a mapping KernelAbstractions doesn't know, which the index functions have to go
# through `expand`, `in` and `linear_index` for
struct TransposedMapping end
Base.@propagate_inbounds function KernelAbstractions.expand(
        ndrange::KernelAbstractions.NDRange{2, B, W, DB, DW, TransposedMapping},
        groupidx::CartesianIndex{2}, idx::CartesianIndex{2}
    ) where {B, W, DB, DW}
    I = (groupidx.I .- 1) .* size(KernelAbstractions.workitems(ndrange)) .+ idx.I
    return CartesianIndex(reverse(I))
end

# like Oceananigans: offsets kept in the dynamic workitems slot, applied by a custom `expand`
struct ItemOffsets{N}
    offsets::NTuple{N, Int}
end
Base.@propagate_inbounds function KernelAbstractions.expand(
        ndrange::KernelAbstractions.NDRange{N, B, W, Nothing, ItemOffsets{N}},
        groupidx::CartesianIndex{N}, idx::CartesianIndex{N}
    ) where {N, B, W}
    I = (groupidx.I .- 1) .* size(KernelAbstractions.workitems(ndrange)) .+ idx.I
    return CartesianIndex(I .+ ndrange.workitems.offsets)
end

@kernel function mapped_indices!(A)
    I = @index(Global, Cartesian)
    @inbounds A[I] = @index(Global, Linear)
end

# host-side logic, independent of the backend
@testset "select_launch" begin
    Testsuite.select_launch_testsuite()
end

@kernel function fill_index!(A)
    I = @index(Global, Linear)
    @inbounds A[I] = I
end

@testset "generic launch" begin
    # workgroup sizes are validated against the kernel's limits
    limit = KernelAbstractions.KI.max_work_group_size(CPU())
    @test_throws ArgumentError fill_index!(CPU())(zeros(Int, 2limit); ndrange = 2limit, workgroupsize = 2limit)

    # iteration spaces with more work-items than an `Int` can count are rejected, instead
    # of launching nothing because the number of workgroups overflowed
    @test_throws ArgumentError fill_index!(CPU())(zeros(Int, 1); ndrange = (2^22, 2^22, 2^22, 1), workgroupsize = 1)
end

# the shared testsuite only covers the launch configuration POCL selects
@testset "POCL launch configurations" begin
    KA = KernelAbstractions
    @testset "custom mapping, $launch" for launch in (nothing, KA.LinearLaunch{Int}(), KA.NDLaunch{Int}())
        # a 7x5 iteration space in 4x4 workgroups, mapped onto a 5x7 ndrange
        kernel = mapped_indices!(CPU(), (4, 4))
        iterspace = KA.NDRange{2, KA.DynamicSize, KA.StaticSize{(4, 4)}}(
            CartesianIndices((2, 2)), nothing, TransposedMapping()
        )
        A = zeros(Int, 5, 7)
        KernelAbstractions.launch_kernel(kernel, launch, CartesianIndices(A), nothing, iterspace, (A,))
        @test A == LinearIndices(A)
    end
    @testset "custom iteration space, $launch" for launch in (nothing, KA.LinearLaunch{Int}(), KA.NDLaunch{Int}())
        # an 8x8 iteration space in 4x4 workgroups, shifted by (1, 2) onto a 7x5 ndrange
        kernel = mapped_indices!(CPU(), (4, 4))
        iterspace = KA.NDRange{2, KA.StaticSize{(2, 2)}, KA.StaticSize{(4, 4)}}(nothing, ItemOffsets((1, 2)))
        ndrange = CartesianIndices((2:8, 3:7))
        A = zeros(Int, 9, 8)
        KernelAbstractions.launch_kernel(kernel, launch, ndrange, nothing, iterspace, (A,))
        @test A[ndrange] == LinearIndices(ndrange)
        A[ndrange] .= 0
        @test all(iszero, A)

        # the launch is chosen from the iteration space
        @test KA.select_launch(kernel, nothing, iterspace) === KA.NDLaunch{Int32}()
    end
    for launch in (nothing, KA.LinearLaunch{Int}(), KA.NDLaunch{Int}())
        function launcher(kernel, args...; ndrange, workgroupsize = nothing)
            ndrange, workgroupsize, iterspace, _ = KA.launch_config(kernel, ndrange, workgroupsize)
            # an N-d launch is limited to three dimensions
            l = launch isa KA.NDLaunch && ndims(iterspace) > 3 ? KA.LinearLaunch{Int}() : launch
            KernelAbstractions.launch_kernel(kernel, l, ndrange, workgroupsize, iterspace, args)
        end
        @testset "$launch" begin
            Testsuite.launch_testsuite(CPU, Array; launcher)
        end
    end
end

@testset "CPU back-end" begin
    Testsuite.testsuite(CPU, "CPU", POCL, Array, POCL.CLDeviceArray)
end

struct NewBackend <: KernelAbstractions.Backend end
@testset "Default host implementation" begin
    backend = NewBackend()

    @test_throws MethodError KernelAbstractions.synchronize(backend)

    @test_throws MethodError KernelAbstractions.allocate(backend, Float32, 1)
    @test_throws MethodError KernelAbstractions.allocate(backend, Float32, (1,))
    @test_throws MethodError KernelAbstractions.allocate(backend, Float32, 1, 2)

    @test_throws MethodError KernelAbstractions.zeros(backend, Float32, 1)
    @test_throws MethodError KernelAbstractions.ones(backend, Float32, 1)

    # conservative capability defaults
    @test KernelAbstractions.supports_atomics(backend) == false
    @test KernelAbstractions.supports_float64(backend) == false

    @test KernelAbstractions.priority!(backend, :high) === nothing
    @test KernelAbstractions.priority!(backend, :normal) === nothing
    @test KernelAbstractions.priority!(backend, :low) === nothing

    @test_throws ErrorException KernelAbstractions.priority!(backend, :middle)

    @test KernelAbstractions.functional(backend) === missing
end

@testset "Deprecated GPU alias" begin
    @test KernelAbstractions.GPU === KernelAbstractions.Backend
    @test CPU() isa KernelAbstractions.GPU
    @test NewBackend <: KernelAbstractions.GPU
end

@testset "Profiling" begin
    RecordingTracer, with_tracer = Testsuite.RecordingTracer, Testsuite.with_tracer
    # nothing is registered unless running under a profiler (or with `JULIA_KA_NVTXT`)
    @test KernelAbstractions.profiling_active() == !isempty(KernelAbstractions.tracers())

    if !KernelAbstractions.profiling_active()
        @testset "inactive" begin
            @test KernelAbstractions.profiling_range_start("label") === nothing
            @test KernelAbstractions.profiling_range_end(nothing) === nothing
            @test profiling_mark("label") === nothing
            # the label isn't evaluated when nobody listens
            @test (@profiling_range error("label") 1 + 2) == 3
        end
    end

    @testset "macro" begin
        @test (@profiling_range "label" 1 + 2) == 3
        @test (@profiling_range "label" domain = "Custom" 1 + 2) == 3
        @test_throws ErrorException @profiling_range "label" error("boom")
        # assignments remain visible, as with `@time`
        @profiling_range "assign" y = 42
        @test y == 42

        # the expression is evaluated once
        count = Ref(0)
        @test (@profiling_range "once" (count[] += 1)) == 1
        @test count[] == 1

        @test_throws ArgumentError macroexpand(@__MODULE__, :(@profiling_range "label" foo = 1 2))
    end

    @testset "registration" begin
        tracer = RecordingTracer()
        with_tracer(tracer) do tracer
            @test KernelAbstractions.profiling_active()
            # registering twice doesn't duplicate
            KernelAbstractions.register_tracer!(tracer)
            @test count(t -> t === tracer, KernelAbstractions.tracers()) == 1
        end
        @test !(tracer in KernelAbstractions.tracers())
    end

    @testset "ranges and markers" begin
        tracer = with_tracer() do tracer
            @test (
                @profiling_range "outer" domain = "Trixi" begin
                    profiling_mark("inside")
                    @profiling_range "inner $(1 + 1)" 7
                end
            ) == 7
            id = KernelAbstractions.profiling_range_start("explicit"; domain = "X")
            KernelAbstractions.profiling_range_end(id)
        end
        @test tracer.events == [
            (:start, "outer", "Trixi"), (:mark, "inside", "KernelAbstractions"),
            (:start, "inner 2", "KernelAbstractions"), (:end, "inner 2"), (:end, "outer"),
            (:start, "explicit", "X"), (:end, "explicit"),
        ]

        # labels fixed in the code are `Symbol`s, which tracers may cache; others `String`s
        @test tracer.types[1:3] == [(Symbol, Symbol), (String, Symbol), (String, Symbol)]
        tracer = with_tracer() do tracer
            Testsuite.profiling_fill!(CPU())(zeros(Float32, 4), 1.0f0; ndrange = 4)
            wait(KernelAbstractions.@spawn CPU() nothing)
        end
        @test all(==((Symbol, Symbol)), tracer.types)
        @test KernelAbstractions.kernel_label(Testsuite.gpu_profiling_fill!) === :profiling_fill!

        # ranges end when the expression throws
        tracer = with_tracer() do tracer
            @test_throws ErrorException @profiling_range "throws" error("boom")
        end
        @test tracer.events == [(:start, "throws", "KernelAbstractions"), (:end, "throws")]

        # ranges end with the tracers they started with
        tracer = KernelAbstractions.register_tracer!(RecordingTracer())
        id = KernelAbstractions.profiling_range_start("open")
        KernelAbstractions.unregister_tracer!(tracer)
        KernelAbstractions.profiling_range_end(id)
        @test tracer.events == [(:start, "open", "KernelAbstractions"), (:end, "open")]

        # from many tasks at once
        tracer = with_tracer() do tracer
            @sync for i in 1:16
                Threads.@spawn @profiling_range "task $i" (yield(); i)
            end
        end
        @test count(e -> e[1] === :start, tracer.events) == 16
        @test count(e -> e[1] === :end, tracer.events) == 16
    end

    @testset "multiple tracers" begin
        a, b = RecordingTracer(), RecordingTracer()
        with_tracer(a) do _
            with_tracer(b) do _
                @profiling_range "both" nothing
            end
        end
        @test a.events == b.events == [(:start, "both", "KernelAbstractions"), (:end, "both")]
    end

    # with a profiler listening
    with_tracer() do _
        Testsuite.profiling_testsuite(CPU, Array)
    end
end

@testset "@profile" begin
    kfill! = Testsuite.profiling_fill!
    A = zeros(Float32, 64)

    results = KernelAbstractions.@profile for i in 1:3
        @profiling_range "step" domain = "Demo" begin
            kfill!(CPU())(A, Float32(i); ndrange = length(A))
            profiling_mark("half")
            kfill!(CPU())(A, Float32(i); ndrange = length(A))
        end
    end
    @test !KernelAbstractions.profiling_active()
    @test all(==(3), A)
    @test count(r -> r.name == "Demo: step", results.ranges) == 3
    @test count(r -> r.name == "profiling_fill!", results.ranges) == 6
    @test length(results.markers) == 3
    @test all(r -> results.start <= r.start <= r.stop <= results.stop, results.ranges)

    # on a backend without timestamps, kernels are timed on the host
    @test count(k -> k.name == "profiling_fill!", results.kernels) == 6
    @test all(k -> k.host_timed && k.device == "POCLBackend 1", results.kernels)
    @test all(k -> results.start <= k.start <= k.stop <= results.stop, results.kernels)

    summary = sprint(show, MIME"text/plain"(), results)
    @test startswith(summary, "Profiled ")
    @test occursin("recording 9 ranges, 3 markers and 6 kernels.", summary)
    lines = split(summary, '\n')
    host = findfirst(==("Host-side activity:"), lines)
    device = findfirst(==("Device-side activity:"), lines)
    @test host !== nothing && device !== nothing && host < device
    @test occursin("Total time", lines[host + 1])
    # sorted by total time
    @test endswith(lines[host + 3], "Demo: step") && endswith(lines[host + 4], "profiling_fill!")
    @test endswith(lines[device + 3], "profiling_fill! *")
    @test any(l -> occursin("timed on the host", l), lines)
    @test any(l -> occursin(r"^ +3  half$", l), lines)

    # without device timing, there are only host ranges
    results = KernelAbstractions.@profile device = false kfill!(CPU())(A, 1.0f0; ndrange = length(A))
    @test isempty(results.kernels) && length(results.ranges) == 1
    @test KernelAbstractions.KI.record_timestamp(NewBackend()) === nothing

    trace = sprint(
        show, MIME"text/plain"(), KernelAbstractions.@profile trace = true begin
            @profiling_range "outer" begin
                profiling_mark("mark")
                @profiling_range "inner" nothing
            end
        end
    )
    lines = split(trace, '\n')
    @test occursin("Duration", lines[3])
    @test endswith(lines[5], "  outer") && endswith(lines[6], "    ◆ mark") && endswith(lines[7], "    inner")

    @test occursin("recording 0 ranges.", sprint(show, MIME"text/plain"(), KernelAbstractions.@profile 1 + 1))

    # launches synchronize their backend only if asked to
    tracer = KernelAbstractions.ProfileTracer(true)
    # only for the tasks it profiles
    @test !KernelAbstractions.synchronizes_launches(tracer)
    @test KernelAbstractions.with(KernelAbstractions.PROFILERS => [tracer]) do
        KernelAbstractions.synchronizes_launches(tracer)
    end
    @test !KernelAbstractions.synchronizes_launches(KernelAbstractions.ProfileTracer(false))
    tracer = KernelAbstractions.ProfileTracer(false, true)
    @test !KernelAbstractions.records_kernels(tracer)
    @test KernelAbstractions.with(KernelAbstractions.PROFILERS => [tracer]) do
        KernelAbstractions.records_kernels(tracer)
    end
    @test !KernelAbstractions.synchronizes_launches(Testsuite.RecordingTracer())
    results = KernelAbstractions.@profile synchronize = false kfill!(CPU())(A, 1.0f0; ndrange = length(A))
    @test only(results.ranges).name == "profiling_fill!"

    # the profiler stops when the expression throws
    @test_throws ErrorException KernelAbstractions.@profile error("boom")
    @test !KernelAbstractions.profiling_active()
    @test_throws ArgumentError macroexpand(@__MODULE__, :(KernelAbstractions.@profile foo = 1 2))

    @testset "tasks" begin
        As = [zeros(Float32, 64) for _ in 1:3]
        work(i) = @profiling_range "task $i" kfill!(CPU())(As[i], 1.0f0; ndrange = length(As[i]))

        # spawned tasks are recorded, and numbered after the profiling task
        results = KernelAbstractions.@profile @profiling_range "parent" begin
            @sync for i in 1:3
                KernelAbstractions.@spawn CPU() work(i)
            end
        end
        @test only(r.task for r in results.ranges if r.name == "parent") == 1
        @test sort([r.task for r in results.ranges if startswith(r.name, "task ")]) == 2:4
        # each kernel range is on the task that launched it
        for i in 1:3
            task = only(r.task for r in results.ranges if r.name == "task $i")
            @test count(r -> r.name == "profiling_fill!" && r.task == task, results.ranges) == 1
        end
        # as is each kernel, on its task's queue
        @test sort([k.task for k in results.kernels]) == 2:4
        # `@spawn` ranges are named after the call site, and belong to the spawned task
        spawns = filter(r -> startswith(r.name, "@spawn runtests.jl:"), results.ranges)
        @test sort([r.task for r in spawns]) == 2:4
        for r in spawns
            child = only(c for c in results.ranges if c.task == r.task && startswith(c.name, "task "))
            @test r.start <= child.start <= child.stop <= r.stop
        end
        named = KernelAbstractions.@profile wait(KernelAbstractions.@spawn CPU() name = "named" nothing)
        @test only(named.ranges).name == "named"
        trace = sprint(show, MIME"text/plain"(), KernelAbstractions.ProfileResults(results.start, results.stop, results.ranges, results.markers, results.kernels, true))
        @test occursin("task 1 (thread ", trace)
        @test occursin("POCLBackend 1, task 2", trace)

        # other tasks aren't
        stop = Threads.Atomic{Bool}(false)
        other = Threads.@spawn while !stop[]
            @profiling_range "unrelated" yield()
        end
        results = KernelAbstractions.@profile for _ in 1:10
            @profiling_range "related" yield()
        end
        stop[] = true
        wait(other)
        @test all(r -> r.name == "related", results.ranges)
        @test length(results.ranges) == 10

        # nor are other profiles, at the same time or nested
        t1 = Threads.@spawn KernelAbstractions.@profile for _ in 1:5
            @profiling_range "one" yield()
        end
        t2 = Threads.@spawn KernelAbstractions.@profile for _ in 1:7
            @profiling_range "two" yield()
        end
        r1, r2 = fetch(t1), fetch(t2)
        @test all(r -> r.name == "one", r1.ranges) && length(r1.ranges) == 5
        @test all(r -> r.name == "two", r2.ranges) && length(r2.ranges) == 7
        local inner
        outer = KernelAbstractions.@profile @profiling_range "outer" begin
            inner = KernelAbstractions.@profile @profiling_range "inner" nothing
        end
        @test sort([r.name for r in outer.ranges]) == ["inner", "outer"]
        @test [r.name for r in inner.ranges] == ["inner"]

        # ranges of tasks that outlive the profile are lost, with a warning
        started, finish = Channel{Nothing}(1), Channel{Nothing}(1)
        local task
        results = @test_logs (:warn, r"1 profiled range still open") KernelAbstractions.@profile begin
            task = Threads.@spawn @profiling_range "outlives" begin
                put!(started, nothing)
                take!(finish)
            end
            take!(started)
        end
        put!(finish, nothing)
        wait(task)
        @test isempty(results.ranges)

        # also for a `@spawn` task that hasn't started yet, as its range starts at `@spawn`
        go = Channel{Nothing}(1)
        results = @test_logs (:warn, r"1 profiled range still open") KernelAbstractions.@profile begin
            task = KernelAbstractions.@spawn CPU() take!(go)
        end
        put!(go, nothing)
        wait(task)
    end

    @test KernelAbstractions.format_time(5) == "5 ns"
    @test KernelAbstractions.format_time(999.7) == "1 µs"
    @test KernelAbstractions.format_time(1.234e6) == "1.23 ms"
    @test KernelAbstractions.format_time(2.5e9) == "2.5 s"
end

@testset "NVTXT" begin
    @test KernelAbstractions.nvtxt_path("1") == "ka-$(getpid()).nvtxt"
    @test KernelAbstractions.nvtxt_path("/tmp/trace-%p.nvtxt") == "/tmp/trace-$(getpid()).nvtxt"

    mktempdir() do dir
        path = joinpath(dir, "trace.nvtxt")
        tracer = KernelAbstractions.NVTXTTracer(path)
        Testsuite.with_tracer(tracer) do _
            @profiling_range "range" nothing
            @profiling_range "say \"hi\"\n" domain = "Trixi" nothing
            profiling_mark("marker")
            Testsuite.profiling_fill!(CPU())(zeros(Float32, 4), 1.0f0; ndrange = 4)
        end
        close(tracer)
        # recording after closing is harmless
        KernelAbstractions.trace_mark(tracer, "late", :KernelAbstractions)

        lines = readlines(path)
        @test lines[1] == "SetFileDisplayName, KernelAbstractions"
        @test "ProcessId = $(getpid())" in lines
        records = filter(l -> startswith(l, "RangeStartEnd, ") || startswith(l, "Marker, "), lines)
        @test length(records) == 4
        r = match(r"^RangeStartEnd, (\d+), (\d+), (\d+), \"range\"$", records[1])
        @test r !== nothing && parse(UInt64, r[1]) <= parse(UInt64, r[2])
        @test endswith(records[2], ", \"Trixi: say 'hi' \"")
        @test match(r"^Marker, \d+, \d+, \"marker\"$", records[3]) !== nothing
        @test endswith(records[4], ", \"profiling_fill!\"")
    end

    # enabled with an environment variable
    mktempdir() do dir
        julia = Cmd(filter(arg -> !startswith(arg, "--code-coverage"), Base.julia_cmd().exec))
        script = """
        using KernelAbstractions
        @profiling_range "from env" nothing
        print(getpid())
        """
        cmd = `$julia --startup-file=no --project=$(Base.active_project()) -e $script`
        env = ("JULIA_KA_NVTXT" => joinpath(dir, "env-%p.nvtxt"),)
        pid = readchomp(setenv(cmd, copy(ENV)..., env...; dir))
        trace = read(joinpath(dir, "env-$pid.nvtxt"), String)
        @test occursin("\"from env\"", trace)
    end
end

import IntelITT, NVTX
@testset "Profiler extensions" begin
    # only registered under the profiler
    itt = Base.get_extension(KernelAbstractions, :IntelITTExt)
    @test isassigned(itt.TRACER) == IntelITT.isactive()
    nvtx = Base.get_extension(KernelAbstractions, :NVTXExt)
    @test isassigned(nvtx.TRACER) == NVTX.isactive()

    # but work without it
    for tracer in (itt.ITTTracer(), nvtx.NVTXTracer())
        Testsuite.with_tracer(tracer) do _
            @test (@profiling_range "range" domain = "Ext" 1) == 1
            @test profiling_mark("mark") === nothing
            Testsuite.profiling_fill!(CPU())(zeros(Float32, 4), 1.0f0; ndrange = 4)
        end
    end
end
