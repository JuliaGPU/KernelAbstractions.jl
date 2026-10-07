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

    # as many threads as Julia has
    @test compute_units(threads = 2) == 2
    # unless PoCL is configured otherwise
    @test compute_units("POCL_CPU_MAX_CU_COUNT" => "5") == 5
    # but KernelAbstractions' variable takes precedence
    @test compute_units("JULIA_KA_CPU_THREADS" => "3") == 3
    @test compute_units("JULIA_KA_CPU_THREADS" => "3", "POCL_MAX_PTHREAD_COUNT" => "5") == 3
    # up to the size of PoCL's device
    @test compute_units("JULIA_KA_CPU_THREADS" => "6", "POCL_MAX_PTHREAD_COUNT" => "5") == 5
end

@testset "POCL atomics descriptor" begin
    # pocl's CPU device supports single- and double-precision addition in global and local
    # memory, 64-bit integer atomics, and no half-precision atomics
    dev = POCL.device()
    atomics = POCL.spirv_atomics(dev)
    @test atomics == POCL.SPIRVAtomics(;
        int64 = true, fadd_f32_global = true, fadd_f32_local = true,
        fadd_f64_global = true, fadd_f64_local = true
    )
    @test POCL.compiler_config(dev).target.atomics == atomics
    # explicitly requested extensions are kept
    config = POCL.compiler_config(dev; extensions = "+SPV_KHR_expect_assume")
    @test config.target.extensions == "+SPV_KHR_expect_assume"
    @test config.target.atomics == atomics
    @test POCL.compiler_config(dev; extensions = nothing).target ==
        POCL.compiler_config(dev).target

    # additions the descriptor doesn't cover become compare-and-swap loops
    function fadd!(L, G)
        @inbounds KernelAbstractions.@atomic L[1] += 1.0f0
        @inbounds KernelAbstractions.@atomic G[1] += 1.0f0
        return
    end
    tt = Tuple{
        POCL.CLDeviceArray{Float32, 1, POCL.AS.Workgroup},
        POCL.CLDeviceArray{Float32, 1, POCL.AS.CrossWorkgroup},
    }
    function spirv(atomics)
        config = POCL.compiler_config(dev; kernel = false, atomics)
        job = POCL.GPUCompiler.CompilerJob(POCL.methodinstance(typeof(fadd!), tt), config)
        return sprint(io -> POCL.GPUCompiler.code_native(io, job))
    end
    for (atomics, native) in (
            (POCL.spirv_atomics(dev), 2),
            (POCL.SPIRVAtomics(; fadd_f32_global = true), 1),
            (POCL.SPIRVAtomics(; fadd_f32_local = true), 1),
            (POCL.SPIRVAtomics(), 0),
        )
        asm = spirv(atomics)
        @test count("OpAtomicFAddEXT", asm) == native
        @test count("OpAtomicCompareExchange", asm) == 2 - native
        @test occursin("SPV_EXT_shader_atomic_float_add", asm) == (native > 0)
    end
end

# 8- and 16-bit atomics are performed on the containing 32-bit word, so they contend with
# the neighbouring values
@kernel function partword_add!(A)
    i = @index(Global, Linear)
    j = (i - 1) % length(A) + 1
    @inbounds KernelAbstractions.@atomic A[j] += one(eltype(A))
end
# atomics on the odd elements only, which must leave the even ones sharing their words intact
@kernel function partword_odd_add!(A)
    i = @index(Global, Linear)
    j = (i - 1) % length(A) + 1
    if isodd(j)
        @inbounds KernelAbstractions.@atomic A[j] += one(eltype(A))
    end
end
# plain stores to the even elements, once no atomics are in flight
@kernel function partword_even_store!(A)
    j = @index(Global, Linear)
    if iseven(j)
        @inbounds A[j] = eltype(A)(j % 64)
    end
end
@kernel function partword_local!(out)
    i = @index(Local, Linear)
    T = @uniform eltype(out)
    lmem = @localmem T (5,)
    if i <= 5
        @inbounds lmem[i] = zero(T)
    end
    @synchronize
    @inbounds KernelAbstractions.@atomic lmem[(i - 1) % 5 + 1] += one(T)
    @synchronize
    if i <= 5
        @inbounds out[i] = lmem[i]
    end
end
@testset "POCL 8- and 16-bit atomics ($T)" for T in (Int8, UInt16, Float16)
    k = 64
    @testset "$n elements" for n in (1, 3, 5, 7, 1023)
        # the arrays that Julia allocates for us; their data being 4-byte aligned is a
        # regression check of what the containing words rely on, not proof of it
        arrays = (
            "allocate" => () -> KernelAbstractions.zeros(CPU(), T, n),
            "resize!" => () -> fill!(resize!(zeros(T, 2), n), zero(T)),
            "copy" => () -> copy(zeros(T, n)),
            "similar" => () -> fill!(similar(zeros(T, 2), n), zero(T)),
        )
        @testset "$name" for (name, alloc) in arrays
            A = alloc()
            @test length(A) == n
            @test UInt(pointer(A)) % 4 == 0
            partword_add!(CPU())(A; ndrange = n * k)
            synchronize(CPU())
            @test all(==(T(k)), A)

            # next to elements that aren't updated, and that are modified by plain stores
            # afterwards (not concurrently, which the containing word rules out)
            B = alloc()
            B .= [isodd(j) ? zero(T) : typemax(T) for j in 1:n]
            partword_odd_add!(CPU())(B; ndrange = n * k)
            synchronize(CPU())
            @test B == [isodd(j) ? T(k) : typemax(T) for j in 1:n]
            partword_even_store!(CPU())(B; ndrange = n)
            synchronize(CPU())
            @test B == [isodd(j) ? T(k) : T(j % 64) for j in 1:n]
        end
    end

    @testset "local memory" begin
        out = zeros(T, 5)
        partword_local!(CPU(), 64)(out; ndrange = 64)
        synchronize(CPU())
        @test out == T[13, 13, 13, 13, 12]
    end
end

# Julia 1.12 checks the bounds of a `StepRange` in 128-bit integers, which the SPIR-V back-end
# cannot lower; this is reached e.g. by indexing a strided view with `--check-bounds=yes`
@testset "POCL bounds checks of a StepRange" begin
    @kernel function steprange_checkbounds!(out, r)
        I = @index(Global, Linear)
        @inbounds out[I] = checkbounds(Bool, r, I - 2)
    end
    @testset "$(typeof(r))" for r in (1:2:8, 9:-3:1, UInt64(1):UInt64(2):UInt64(8), Int32(1):Int32(2):Int32(8))
        n = Int(length(r))
        out = zeros(Bool, n + 3)
        steprange_checkbounds!(CPU())(out, r; ndrange = length(out))
        synchronize(CPU())
        @test out == [checkbounds(Bool, r, i) for i in -1:(n + 1)]
    end

    A = zeros(Int, 8, 10)
    v = view(A, 1:2:8, 2:2:10)
    @kernel function strided_view_fill!(v)
        I = @index(Global, Cartesian)
        v[I] = 1
    end
    strided_view_fill!(CPU())(v; ndrange = size(v))
    synchronize(CPU())
    ref = zeros(Int, 8, 10)
    ref[1:2:8, 2:2:10] .= 1
    @test A == ref
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
