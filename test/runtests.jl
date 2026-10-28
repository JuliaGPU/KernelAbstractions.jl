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
    vars = ("JULIA_KA_CPU_THREADS", POCL.cl.pocl_thread_variables...)
    function query(property, env...; threads = 2)
        script = "using KernelAbstractions: POCL; print(POCL.$property)"
        cmd = `$julia --startup-file=no --threads=$threads
            --project=$(Base.active_project()) -e $script`
        cmd_env = filter(kv -> !(first(kv) in vars), copy(ENV))
        return parse(Int, readchomp(setenv(cmd, cmd_env..., env...)))
    end
    compute_units(env...; kwargs...) = query("device().max_compute_units", env...; kwargs...)

    # as many threads as Julia has
    @test compute_units(threads = 2) == 2
    # unless PoCL is configured otherwise
    @test compute_units("POCL_CPU_MAX_CU_COUNT" => "5") == 5
    # but KernelAbstractions' variable takes precedence
    @test compute_units("JULIA_KA_CPU_THREADS" => "3") == 3
    @test compute_units("JULIA_KA_CPU_THREADS" => "3", "POCL_MAX_PTHREAD_COUNT" => "5") == 3
    # up to the size of PoCL's device
    @test compute_units("JULIA_KA_CPU_THREADS" => "6", "POCL_MAX_PTHREAD_COUNT" => "5") == 5

    # host launches cannot exceed Julia's default pool size
    host_threads(env...) = query("device_limits().compute_units", env...)
    @test host_threads() == 2
    @test host_threads("JULIA_KA_CPU_THREADS" => "1") == 1
    @test host_threads("JULIA_KA_CPU_THREADS" => "3") == 2
    @test host_threads("POCL_CPU_MAX_CU_COUNT" => "5") == 2
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
    # explicitly requested extensions are kept, in addition to the default ones and the one
    # for vector shuffles
    @test POCL.compiler_config(dev).target.extensions ==
        "+SPV_KHR_no_integer_wrap_decoration,+SPV_INTEL_subgroups"
    config = POCL.compiler_config(dev; extensions = "+SPV_KHR_expect_assume")
    @test config.target.extensions ==
        "+SPV_KHR_no_integer_wrap_decoration,+SPV_INTEL_subgroups,+SPV_KHR_expect_assume"
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

@testset "POCL reinterpret" begin
    @kernel function reinterpret_copy!(out, a)
        I = @index(Global, Linear)
        @inbounds out[I] = reinterpret(eltype(out), a)[I]
    end
    a = Float32[1, 2, 3, 4]
    @testset "$T" for T in (UInt32, UInt16)
        out = zeros(T, length(reinterpret(T, a)))
        reinterpret_copy!(CPU())(out, a; ndrange = length(out))
        synchronize(CPU())
        @test out == reinterpret(T, a)
    end

    # like on the host, arrays of non-bitstypes can't be reinterpreted
    out = zeros(UInt8, 1)
    @test_throws POCL.KernelException redirect_stdout(devnull) do
        reinterpret_copy!(CPU())(out, Union{Int8, UInt8}[Int8(-1)]; ndrange = 1)
        synchronize(CPU())
    end
end

# `@Const` arguments are constified by the launch, using the mask of the kernel method
# that is called
@kernel function const_seen!(out, @Const(B::AbstractVector))
    I = @index(Global, Linear)
    @inbounds out[I] = B isa POCL.Const
end
@kernel function const_seen!(out, B::Number)
    I = @index(Global, Linear)
    @inbounds out[I] = B isa POCL.Const
end
@kernel function const_seen_vararg!(out, @Const(B), Cs...)
    I = @index(Global, Linear)
    @inbounds out[I] = (B isa POCL.Const) && !any(C -> C isa POCL.Const, Cs)
end
# the defaults are evaluated on the device, not when looking up which arguments are
# `@Const`, and can refer to the ones before them
@kernel function const_seen_default!(
        out, @Const(B), C = B isa POCL.Const || error("evaluated on the host"), D = C
    )
    I = @index(Global, Linear)
    @inbounds out[I] = C && D
end
@kernel generated = true function const_seen_generated!(out, @Const(B))
    I = @index(Global, Linear)
    @inbounds out[I] = $(B <: POCL.Const)
end
@testset "POCL @Const arguments" begin
    B = ones(Float32, 4)
    @testset "$name" for (name, kernel, args, expected) in (
            ("marked method", const_seen!, (B,), true),
            ("unmarked method", const_seen!, (1.0f0,), false),
            ("varargs", const_seen_vararg!, (B, B, B), true),
            ("default arguments", const_seen_default!, (B,), true),
            ("default argument given", const_seen_default!, (B, true), true),
            ("generated", const_seen_generated!, (B,), true),
        )
        out = zeros(Bool, 4)
        kernel(CPU())(out, args...; ndrange = length(out))
        synchronize(CPU())
        @test all(==(expected), out)
    end
    @test_throws "`@Const` cannot be applied to varargs" macroexpand(
        @__MODULE__, :(@kernel function const_varargs!(out, @Const(Bs...)) end)
    )
end

# the RNG state takes two UInt32s of local memory per sub-group
@testset "POCL device RNG: local memory" begin
    rng_kernel(A) = (A[POCL.get_global_id()] = rand(Float32); return)
    A = zeros(Float32, 32)
    kernel = POCL.clfunction(rng_kernel, Tuple{typeof(POCL.clconvert(A))})
    @test kernel.rng_state
    kernel(A; global_size = 32, local_size = 32)
    num_sub_groups = POCL.cl.sub_group_info(kernel.fun, POCL.device(), Csize_t[32]).sub_group_count
    local_mem = POCL.cl.work_group_info(kernel.fun, POCL.device()).local_mem_size
    @test local_mem == 2 * sizeof(UInt32) * num_sub_groups
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

# the sub-group width kernels are compiled for is the one `KI.sub_group_size` reports
# (PoCL folds `get_max_sub_group_size` to that constant itself, see pocl/pocl#2375)
function max_sub_group_size_kernel(out)
    out[1] = KernelAbstractions.KernelInterface.get_max_sub_group_size()
    return
end
@testset "POCL sub-group width" begin
    out = zeros(Int, 1)
    @opencl max_sub_group_size_kernel(out)
    @test out[1] == POCL.device_limits().sub_group_size
end

# the native sub-group collectives are used where they have Julia's semantics
function sub_group_reduce_kernel(out, x)
    i = KernelAbstractions.KernelInterface.get_global_id().x
    out[i] = KernelAbstractions.KernelInterface.sub_group_reduce(+, x[i])
    return
end
function sub_group_exclusive_scan_kernel(out, x)
    i = KernelAbstractions.KernelInterface.get_global_id().x
    out[i] = KernelAbstractions.KernelInterface.sub_group_exclusive_scan(+, x[i], eltype(x)(7))
    return
end
@testset "POCL native sub-group collectives" begin
    W = POCL.device_limits().sub_group_size
    for T in (Int32, Float32)
        x, out = ones(T, W), zeros(T, W)
        ir = sprint() do io
            @device_code_llvm io = io debuginfo = :none @opencl local_size = W global_size = W sub_group_reduce_kernel(out, x)
        end
        @test all(==(W), out)
        @test occursin("sub_group_reduce_add", ir)

        ir = sprint() do io
            @device_code_llvm io = io debuginfo = :none @opencl local_size = W global_size = W sub_group_exclusive_scan_kernel(out, x)
        end
        @test out == 7 .+ (0:(W - 1))
        @test occursin("sub_group_scan_exclusive_add", ir)
    end
end

# structs are packed into words, which are exchanged with as few vector shuffles as possible
function packed_shuffle_kernel(out, x)
    KI = KernelAbstractions.KernelInterface
    i = KI.get_global_id().x
    out[i] = KI.shfl(x[i], KI.get_sub_group_local_id() % KI.get_sub_group_size() + 1)
    return
end
@testset "POCL vector shuffles of packed structs" begin
    W = POCL.device_limits().sub_group_size
    for n in (1, 2, 5, 16, 17, 18, 33)
        x = [ntuple(k -> UInt32(1000i + k), n) for i in 1:2W]
        out = similar(x)
        ir = sprint() do io
            @device_code_llvm io = io debuginfo = :none @opencl local_size = 2W global_size = 2W packed_shuffle_kernel(out, x)
        end
        @test out == [x[fld(i - 1, W) * W + mod1(i - fld(i - 1, W) * W + 1, W)] for i in 1:2W]
        # vectors of up to 16 words, and a single word as a scalar
        @test count(r"call [^\n]*SubgroupShuffleINTEL", ir) == fld(n, 16) + (n % 16 >= 2)
    end
end

# `@groupreduce` reduces sub-groups with PoCL's native collectives
@kernel function groupreduce_codegen!(out, @Const(x))
    i = @index(Global, Linear)
    res = @groupreduce(+, x[i], zero(eltype(out)))
    out[i] = res
end
@testset "POCL @groupreduce with sub-groups" begin
    x = ones(Int32, 100)
    out = zeros(Int32, 100)
    ir = sprint() do io
        @device_code_llvm io = io debuginfo = :none groupreduce_codegen!(POCLBackend(), 64)(out, x; ndrange = 100)
    end
    @test out == [fill(64, 64); fill(36, 36)]
    @test occursin("sub_group_reduce_add", ir)
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

# launching a compiled kernel doesn't allocate
@testset "POCL launch allocations" begin
    mod = @eval module $(gensym())
    import KernelAbstractions, KernelAbstractions.POCL
    using KernelAbstractions: @kernel, @index
    kernel(A) = (A[POCL.get_global_id()] = 1; return)
    compiler_config(dev) = POCL.compiler_config(dev; name = nothing)
    # `tt` is only specialized on as a type parameter
    clfunction(::Type{TT}) where {TT} = POCL.clfunction(kernel, TT; name = nothing)
    kernel_function(::Type{TT}) where {TT} =
        KernelAbstractions.KernelInterface.kernel_function(POCL.POCLBackend(), kernel, TT)
    function launch_tuple(kernel, A, info)
        event = POCL.launch_tuple(kernel, (A,), info; global_size = (4,), local_size = (4,))
        wait(event)
        POCL.cl.clReleaseEvent(event)
    end
    @kernel function ka_kernel!(A, n)
        I = @index(Global, Linear)
        x = A[I]
        for _ in 1:n
            x = sin(x)
        end
        @inbounds A[I] = x
    end
    launch(kernel, A, n) = kernel(A, n; ndrange = length(A))
    end
    # with `args...`, Julia 1.11 and older box the result
    allocated(f::F, args::Vararg{Any, N}) where {F, N} = (f(args...); @allocated f(args...))

    # these allocate on Julia 1.10, and those that look up compiled code in GPUCompiler on 1.11
    skip = VERSION < v"1.11"
    skip_lookup = VERSION < v"1.12"
    tt = Tuple{typeof(POCL.clconvert(zeros(Int, 4)))}
    @test allocated(mod.compiler_config, POCL.device()) == 0 skip = skip
    @test allocated(mod.clfunction, tt) == 0 skip = skip_lookup
    @test allocated(mod.kernel_function, tt) == 0 skip = skip_lookup

    # neither does waiting for the kernel, also when it takes long enough for the wait to
    # involve a completion callback (except on 1.11, and to shield it from task cancellation
    # on 1.14+)
    skip_wait = VERSION < v"1.12" || isdefined(Base, :CANCEL_TOKEN)
    A = zeros(Int, 4)
    info = POCL.exception_info()
    GC.@preserve info begin
        info_ptr = Base.unsafe_convert(Ptr{POCL.ExceptionInfo_st}, info)
        kernel = mod.clfunction(tt)
        mod.launch_tuple(kernel, A, info_ptr)
        @test all(==(1), A)
        @test allocated(mod.launch_tuple, kernel, A, info_ptr) == 0 skip = skip_wait
    end
    B = ones(4)
    kernel = mod.ka_kernel!(CPU(), 4)
    for n in (1, 100_000)
        mod.launch(kernel, B, n)
        @test allocated(mod.launch, kernel, B, n) == 0 skip = skip_wait
    end

    # nor does checking what the device supports, which launches of collectives do
    for T in (Float16, Float32, Float64)
        @test allocated(KernelAbstractions.KernelInterface.supports_shuffle, CPU(), T) == 0
    end
    @test allocated(KernelAbstractions.supports_float64, CPU()) == 0
end

@testset "POCL long host launches" begin
    mod = @eval module $(gensym())
    using KernelAbstractions.POCL
    function kernel(A, n)
        i = POCL.get_global_id()
        x = A[i]
        for _ in 1:n
            x = sin(x)
        end
        A[i] = x
        return
    end
    end
    A = ones(4)
    kernel = POCL.clfunction(mod.kernel, POCL.argument_types((A, 1)))
    sizes = (; global_size = (4, 1, 1), local_size = (4, 1, 1))
    h = POCL.host_kernel(kernel, sizes, nothing)
    @test h !== nothing
    # force the long path even on fast machines
    function launch_long(kernel, h, A, n, sizes)
        Base.@atomic :monotonic h.last_duration = typemax(UInt64)
        kernel(A, n; sizes...)
    end
    allocated(f::F, args...) where {F} = (f(args...); @allocated f(args...))
    # shielding an active cancellation scope allocates on Julia 1.14+
    bytes = @static if isdefined(Base, :CANCEL_TOKEN)
        Base.ScopedValues.with(Base.CANCEL_TOKEN => nothing) do
            allocated(launch_long, kernel, h, A, 1, sizes)
        end
    else
        allocated(launch_long, kernel, h, A, 1, sizes)
    end
    @test bytes == 0 skip = VERSION < v"1.12"

    # a long launch lets a queued task run afterwards; back-to-back launches let timers
    # fire too
    progressed = Ref(false)
    task = @async (progressed[] = true)
    launch_long(kernel, h, A, 1, sizes)
    @test progressed[]
    wait(task)
    woke = Threads.Atomic{Bool}(false)
    task = @async (sleep(0.01); woke[] = true)
    t0 = time()
    while !woke[] && time() - t0 < 10
        launch_long(kernel, h, A, 1, sizes)
    end
    @test woke[]
    wait(task)
    # also with launches from tasks on all threads, of which only one runs timers
    stop = Threads.Atomic{Bool}(false)
    woke = Threads.Atomic{Bool}(false)
    timer = Timer(_ -> (woke[] = true), 0.01)
    tasks = map(2:Threads.nthreads()) do _
        Threads.@spawn begin
            B = ones(4)
            while !stop[]
                launch_long(kernel, h, B, 1, sizes)
            end
        end
    end
    t0 = time()
    while !woke[] && time() - t0 < 10
        launch_long(kernel, h, A, 1, sizes)
    end
    stop[] = true
    foreach(wait, tasks)
    close(timer)
    @test woke[]

    # concurrent callers need separate argument storage
    @sync for _ in 1:4
        Threads.@spawn begin
            B = ones(4)
            for _ in 1:3
                launch_long(kernel, h, B, 1000, sizes)
            end
            expected = 1.0
            for _ in 1:3000
                expected = sin(expected)
            end
            @test all(x -> isapprox(x, expected), B)
        end
    end
end

@testset "POCL small host launches" begin
    if POCL.device_limits().compute_units > 1
        mod = @eval module $(gensym())
        using KernelAbstractions.POCL
        function kernel(A, n)
            i = POCL.get_global_id()
            x = A[i]
            for _ in 1:n
                x = sin(x)
            end
            A[i] = x
            return
        end
        end
        s = POCL.session()
        nt = s.limits.compute_units
        A = ones(nt)
        kernel = POCL.clfunction(mod.kernel, POCL.argument_types((A, 1)))
        sizes = (; global_size = (nt, 1, 1), local_size = (1, 1, 1))
        h = POCL.host_kernel(kernel, sizes, nothing)
        @test h !== nothing
        Base.@atomic h.serial_groups_a = 0
        Base.@atomic h.serial_groups_b = 0
        # small launches stay serial unless both recent estimates are slow
        @test POCL.host_schedule(h, s, nt, false) == (1, false)
        kernel(A, 10^5; sizes...)
        @test POCL.host_schedule(h, s, nt, false) == (1, false)
        kernel(A, 10^5; sizes...)
        @test POCL.host_schedule(h, s, nt, false)[1] == nt
        # team runs also estimate serial duration; sleeping threads can delay the first runs
        for _ in 1:5
            kernel(A, 0; sizes...)
        end
        @test POCL.host_schedule(h, s, nt, false) == (1, false)
        # large grids select the full team
        @test POCL.host_schedule(h, s, 10^6, false)[1] == nt
        # estimates are per grid: a short one-group launch must not make a heavy grid run
        # serially
        kernel(A, 10^5; sizes...)
        kernel(A, 10^5; sizes...)
        kernel(A, 0; global_size = (1, 1, 1), local_size = (1, 1, 1))
        @test POCL.host_schedule(h, s, nt, false)[1] == nt

        # test team and stealing limits without executing oversized grids
        tmax, smax = POCL.host_team_max_groups, POCL.host_steal_max_groups
        @test POCL.host_schedule(h, s, tmax, false) == (nt, tmax <= smax)
        @test POCL.host_schedule(h, s, tmax + 1, false) == (1, false)
        @test POCL.host_schedule(h, s, tmax + 1, true) == (1, false)
        @test POCL.host_schedule(h, s, typemax(Int), true) == (1, false)
        @test POCL.host_schedule(h, s, min(smax, tmax), true) == (nt, true)
        if smax < tmax
            @test POCL.host_schedule(h, s, smax + 1, true) == (nt, false)
            @test POCL.host_schedule(h, s, tmax, true) == (nt, false)
        end
    end
end

@testset "POCL host launch schedule" begin
    if POCL.device_limits().compute_units > 1
        mod = @eval module $(gensym())
        using KernelAbstractions.POCL
        function kernel(A)
            i = POCL.get_global_id()
            A[i] += 1
            return
        end
        end
        s = POCL.session()
        nt = s.limits.compute_units
        A = zeros(nt)
        kernel = POCL.clfunction(mod.kernel, POCL.argument_types((A,)))
        sizes = (; global_size = (nt, 1, 1), local_size = (1, 1, 1))
        h = POCL.host_kernel(kernel, sizes, nothing)
        @test h !== nothing
        function last!(groups, duration)
            # (heavy enough to run on a team, with nt and nt + 1 work-groups)
            for g in (nt + 1, nt)
                POCL.record_serial_sample(h, g, typemax(UInt64))
                POCL.record_serial_sample(h, g, typemax(UInt64))
            end
            Base.@atomic h.last_groups = groups
            Base.@atomic h.last_duration = duration
            return
        end

        # use static blocks after a short launch of the same grid; otherwise allow stealing
        last!(nt, 0)
        @test POCL.host_schedule(h, s, nt, false) == (nt, false)
        @test POCL.host_schedule(h, s, nt + 1, false) == (nt, true)
        @test POCL.host_schedule(h, s, nt, true) == (nt, true)
        last!(0, 0)
        @test POCL.host_schedule(h, s, nt, false) == (nt, true)

        # launches remember how many work-groups they had, and how long they took
        last!(0, 0)
        kernel(A; sizes...)
        @test (Base.@atomic h.last_groups) == nt
        @test (Base.@atomic h.last_duration) < typemax(UInt64)
        kernel(A; sizes...)
        @test A == fill(2, nt)

        # a short launch on a team lets a queued task run, and such launches back to back
        # let a timer fire
        function short_team!()
            last!(nt, 0)
            @assert POCL.host_schedule(h, s, nt, false) == (nt, false)
            kernel(A; sizes...)
        end
        progressed = Ref(false)
        task = @async (progressed[] = true)
        short_team!()
        @test progressed[]
        wait(task)
        woke = Threads.Atomic{Bool}(false)
        task = @async (sleep(0.01); woke[] = true)
        t0 = time()
        while !woke[] && time() - t0 < 10
            short_team!()
        end
        @test woke[]
        wait(task)
    end
end

@testset "POCL host launches on a team" begin
    if POCL.device_limits().compute_units > 1
        @kernel function team_axpy!(y, a, @Const(x))
            i = @index(Global)
            @inbounds y[i] = a * x[i] + y[i]
        end
        @kernel function team_spin!(A, n)
            i = @index(Global)
            v = A[i]
            for _ in 1:n
                v = sin(v)
            end
            A[i] = v
        end
        @kernel function team_exception!(A, i)
            I = @index(Global, Linear)
            if I == i
                A[0] = 1
            end
        end
        # serial execution must produce tickets in work-group order
        @kernel function team_order!(T, counter)
            if @index(Local, Linear) == 1
                @inbounds T[@index(Group, Linear)] = last(KernelAbstractions.@atomic counter[1] += 1)
            end
        end
        # hold the team in group 1 until `release` or `n` checks. `entered` marks its start;
        # `after` marks group `later`, which cannot start during the hold in a serial
        # launch.
        @kernel function team_hold!(entered, after, release, timedout, n, later)
            if @index(Local, Linear) == 1
                g = @index(Group, Linear)
                g == later && (KernelAbstractions.@atomic after[1] = 1)
                if g == 1
                    KernelAbstractions.@atomic entered[1] = 1
                    i = 0
                    while (KernelAbstractions.@atomic release[1]) == 0 && i < n
                        i += 1
                    end
                    i == n && (KernelAbstractions.@atomic timedout[1] = 1)
                end
            end
        end
        # enough work-items to select a team
        n = 64 * POCL.host_kernel_serial_items
        x = rand(Float32, n)
        launch!(y) = team_axpy!(CPU(), 256)(y, 2.0f0, x; ndrange = n)
        launch!(zeros(Float32, n))
        quietly(f) = redirect_stdout(f, devnull)

        # nested parallel launches run serially (see `ThreadTeams.foreach_range`)
        ngroups = n ÷ 64
        function order!(T)
            counter = [0]
            team_order!(CPU(), 64)(T, counter; ndrange = n)
            return T
        end
        order!(zeros(Int, ngroups))
        Ts = [zeros(Int, ngroups) for _ in 1:4]
        Threads.@threads :static for i in 1:4
            order!(Ts[i])
        end
        @test all(==(1:ngroups), Ts)

        # a busy team also forces serial execution. wait for both the held group and the
        # last block to start. the hold may occupy any thread, so let the checking task
        # migrate; always release the hold.
        load(A) = Core.Intrinsics.atomic_pointerref(pointer(A), :acquire)
        entered, after, release, timedout = [0], [0], [0], [0]
        nt = min(ngroups, POCL.device_limits().compute_units)
        later = (nt - 1) * (ngroups ÷ nt) + min(nt - 1, ngroups % nt) + 1
        task = Threads.@spawn team_hold!(CPU(), 64)(entered, after, release, timedout, 10^10, later; ndrange = n)
        check = Threads.@spawn try
            t0 = time()
            while !(load(entered) == 1 && load(after) == 1) && time() - t0 < 30
                yield()
            end
            T = order!(zeros(Int, ngroups))
            load(entered) == 1 && load(after) == 1 && load(timedout) == 0, T
        finally
            Core.Intrinsics.atomic_pointerset(pointer(release), 1, :release)
        end
        held, T = fetch(check)
        wait(task)
        @test held
        @test timedout[1] == 0
        @test T == 1:ngroups

        A = ones(Float32, n)
        team_spin!(CPU(), 256)(A, 1000; ndrange = n)
        expected = foldl((x, _) -> sin(x), 1:1000; init = 1.0f0)
        @test all(a -> isapprox(a, expected; rtol = 1.0e-4), A)

        # report exceptions from any thread without breaking later launches
        A = zeros(Float32, n)
        for i in (1, n ÷ 2, n)
            @test_throws POCL.KernelException quietly(() -> team_exception!(CPU(), 256)(A, i; ndrange = n))
        end
        team_exception!(CPU(), 256)(A, 0; ndrange = n)
        y = zeros(Float32, n)
        launch!(y)
        @test y == 2 .* x
    end
end

# static placement matches `Threads.@threads :static`, regardless of the launching task
@testset "POCL host launch placement" begin
    nt = POCL.device_limits().compute_units
    if nt > 1
        mod = @eval module $(gensym())
        using KernelAbstractions.POCL
        # a work-group's local memory lies in the scratch area of the thread that runs it
        function where_kernel(T)
            mem = POCL.emit_localmemory(UInt8, Val(1))
            if POCL.get_local_id(1) == 1
                @inbounds T[POCL.get_group_id(1)] = reinterpret(UInt, mem)
            end
            return
        end
        end
        function thread_of(address)
            for (tid, s) in enumerate(POCL.host_scratch.slots)
                UInt(s.ptr) <= address < UInt(s.ptr) + s.size && return tid
            end
            return 0
        end
        # enough work-items to select a team
        items = 64
        groups = cld(4 * POCL.host_kernel_serial_items, items) * nt + 3
        T = zeros(UInt, groups)
        k = POCL.clfunction(mod.where_kernel, POCL.argument_types((T,)))
        sizes = (; global_size = (groups * items, 1, 1), local_size = (items, 1, 1))
        h = POCL.host_kernel(k, sizes, nothing)
        @test h !== nothing

        # expected static placement
        tids = Threads.threadpooltids(:default)
        q, r = divrem(groups, nt)
        block(g) = g <= r * (q + 1) ? cld(g, q + 1) : r + cld(g - r * (q + 1), q)
        if nt == Threads.nthreads(:default)
            tt = zeros(Int, groups)
            Threads.@threads :static for g in 1:groups
                tt[g] = Threads.threadid()
            end
            @test tt == tids[block.(1:groups)]
        end

        # count misplaced blocks across `n` launches, excluding whole blocks taken over by
        # the dispatcher when their assigned thread starts late. `stolen_chunks()` counts
        # these takeovers; `dispatcher` identifies the thread when known.
        function misplaced(launch!, dispatcher = nothing; n = 20)
            count = 0
            for _ in 1:n
                fill!(T, 0)
                rescued = POCL.ThreadTeams.stolen_chunks()
                launch!()
                rescued = POCL.ThreadTeams.stolen_chunks() - rescued
                ran = thread_of.(T)
                @test all(!=(0), ran)
                moved = 0
                for b in 1:nt
                    on = unique(ran[filter(g -> block(g) == b, 1:groups)])
                    on == [tids[b]] && continue
                    moved += 1
                    length(on) == 1 && something(dispatcher, on[1]) == on[1] || (count += 1)
                end
                count += max(0, moved - rescued)
            end
            return count
        end
        # force static team scheduling
        function static!(h)
            Base.@atomic :monotonic h.last_duration = 0
            Base.@atomic :monotonic h.last_groups = groups
            return
        end
        @test misplaced(() -> (static!(h); k(T; sizes...)), Threads.threadid()) == 0

        # construct a launch as `host_dispatch` would (one argument, no RNG), with stealing
        # disabled
        function launch_directly(run)
            info = Ref(POCL.ExceptionInfo_st())
            num_groups = Ref(Csize_t.((groups, 1, 1)))
            vals = Ref((POCL.kernel_state(false, C_NULL), POCL.clconvert(T)))
            GC.@preserve T info num_groups vals begin
                info_ptr = Base.unsafe_convert(Ptr{POCL.ExceptionInfo_st}, info)
                vals[] = (POCL.kernel_state(false, info_ptr), vals[][2])
                base = Ptr{Cvoid}(Base.unsafe_convert(Ptr{typeof(vals[])}, vals))
                tt = typeof(vals[])
                ptrs = Ref((base + fieldoffset(tt, 1), base + fieldoffset(tt, 2), C_NULL, C_NULL))
                GC.@preserve ptrs begin
                    launch = POCL.HostLaunch(
                        h, Ptr{Ptr{Cvoid}}(Base.unsafe_convert(Ptr{typeof(ptrs[])}, ptrs)),
                        Ptr{Csize_t}(Base.unsafe_convert(Ptr{NTuple{3, Csize_t}}, num_groups)),
                        groups, nt, false, info_ptr
                    )
                    run(launch)
                end
            end
            @test info[].status == 0
            return
        end
        @test misplaced(() -> launch_directly(l -> l()), Threads.threadid()) == 0
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

# `clfunction` reuses the kernel it returned last for a call site
@testset "POCL kernel reuse" begin
    mod = @eval module $(gensym())
    import KernelAbstractions.POCL
    @noinline helper(x) = x + 1
    kernel(A, x::Number) = (A[1] = helper(x); return)
    launch(A, x) = (POCL.@opencl kernel(A, x); A[1])
    struct Constant
        x::Int
    end
    (c::Constant)(A) = (A[1] = c.x; return)
    launch_callable(f, A) = (POCL.@opencl f(A); A[1])
    end

    count() = POCL.compilations[]
    launch(A, x) = Base.invokelatest(mod.launch, A, x)
    A = zeros(Int, 1)
    @test launch(A, 1) == 2
    n = count()
    @test launch(A, 2) == 3
    @test count() == n

    # defining unrelated methods changes the world, but not the kernel
    @eval mod unrelated() = nothing
    @test launch(A, 2) == 3
    @test count() == n

    # redefining a callee recompiles
    @eval mod @noinline helper(x) = x + 10
    @test launch(A, 2) == 12
    @test count() == n + 1

    # a more specific method of the kernel is used, although the code of the other one is
    # still valid
    world = Base.get_world_counter()
    @eval mod kernel(A, x::Int) = (A[1] = -x; return)
    @test launch(A, 2) == -2
    @test count() == n + 2

    # launches in an older world use the methods of that world
    @test Base.invoke_in_world(world, mod.launch, A, 2) == 12
    @test launch(A, 2) == -2
    @test count() == n + 2

    # callables of the same type are launched with their own values
    @test Base.invokelatest(mod.launch_callable, mod.Constant(1), A) == 1
    @test Base.invokelatest(mod.launch_callable, mod.Constant(2), A) == 2
end

@testset "POCL session" begin
    # in a fresh process, to check initialization, and with threads, which the tests
    # usually don't have
    julia = Cmd(filter(arg -> !startswith(arg, "--code-coverage"), Base.julia_cmd().exec))
    # disable Julia 1.12+'s interactive thread so thread 1 receives SIGINT during team
    # launches
    threads = VERSION >= v"1.12" ? "4,0" : "4"
    cmd = `$julia --startup-file=no --threads=$threads --project=$(Base.active_project())
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

struct ZeroOnePair
    a::Int16
    b::Float32
end
Base.zero(::Type{ZeroOnePair}) = ZeroOnePair(0, 0.0f0)
Base.one(::Type{ZeroOnePair}) = ZeroOnePair(1, 1.0f0)

# large arrays are filled with a kernel, small ones from the host
@testset "POCL zeros/ones of large arrays" begin
    threshold = POCL.POCLKernels.KERNEL_FILL_BYTES
    for T in (Float32, Int8, ComplexF64, ZeroOnePair)
        n = cld(threshold, sizeof(T)) + 3
        for dims in ((n,), (cld(n, 7), 7), (1,))
            Z = KernelAbstractions.zeros(POCLBackend(), T, dims)
            @test Z isa Array{T, length(dims)} && size(Z) == dims
            @test all(==(zero(T)), Z)

            O = KernelAbstractions.ones(POCLBackend(), T, dims...)
            @test O isa Array{T, length(dims)} && size(O) == dims
            @test all(==(one(T)), O)
        end
    end
end

# large copies use a kernel, small ones the host, as do overlapping ones
@testset "POCL copyto!" begin
    threshold = POCL.POCLKernels.KERNEL_FILL_BYTES
    values = (Float32 => Float32, Union{Missing, Int32} => i -> isodd(i) ? Int32(i) : missing)
    for (T, value) in values
        @testset "$T, $m elements" for m in (16, cld(threshold, 4) + 3)
            x = T[value(i) for i in 1:m]
            y = Array{T}(undef, m)
            @test KernelAbstractions.copyto!(POCLBackend(), y, x) === y
            @test isequal(y, x)

            z = copy(x)
            KernelAbstractions.copyto!(POCLBackend(), view(z, 2:m), view(z, 1:(m - 1)))
            @test isequal(z, [x[1]; x[1:(m - 1)]])
        end
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

@testset "select_launch on PoCL" begin
    # tuned launches are folded for any number of work-items in the first three dimensions
    kernel = mapped_indices!(CPU())
    _, wgs, iterspace, _ = KernelAbstractions.launch_config(kernel, (1, 1, 1, 3000), nothing)
    @test KernelAbstractions.select_launch(kernel, wgs, iterspace) === KernelAbstractions.NDLaunch{Int}()
end

@testset "POCL N-d work-group sizes" begin
    # tuned work-groups shrink to divisors of the ndrange where possible, avoiding partial groups
    wgs(ndrange) = KernelAbstractions.launch_workgroupsize(
        CPU(), KernelAbstractions.NDLaunch{Int}(), 4096, ndrange
    )
    # (searching the work-group counts)
    @test wgs((126, 126, 126)) == (126, 21, 1)
    @test wgs((100, 100)) == (100, 25)
    @test wgs((5000, 10)) == (2500, 1)
    # (searching the sizes)
    @test wgs((1024, 1002)) == (1024, 3)
    @test wgs((64, 64, 64)) == (64, 64, 1)
    # shrink to a divisor retaining at least half of each dimension, or to a quarter otherwise
    @test wgs((127, 127, 127)) == (127, 8, 1)
    @test wgs((1024, 1001)) == (1024, 1)
    @test wgs((100003,)) == (1024,)
    @test wgs((0, 7)) == (1, 7)

    # an explicit workgroup size is kept
    @kernel function groupsize!(A)
        if @index(Global, Linear) == 1
            A[1], A[2] = @groupsize()
        end
    end
    A = zeros(Int, 2)
    groupsize!(CPU())(A; ndrange = (126, 126), workgroupsize = (126, 32))
    KernelAbstractions.synchronize(CPU())
    @test A == [126, 32]
    # report the kernel's total-size limit even for a linearized launch
    n = KernelAbstractions.KernelInterface.max_work_group_size(CPU()) ÷ 64 + 1
    @test_throws "work-items" groupsize!(CPU(), (64, n))(A; ndrange = (64, n, 2))
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

# whether the launch runs a kernel that checks its indices
@kernel function bounds_checked!(A)
    I = @index(Global, Linear)
    @inbounds A[I] = KernelAbstractions.__dynamic_checkbounds(__ctx__)
end

@kernel function tuned_bounds_checked!(A)
    I = @index(Global, Linear)
    @inbounds A[I] = KernelAbstractions.__dynamic_checkbounds(__ctx__)
end

@kernel function partial_bounds_checked!(A)
    I = @index(Global, Linear)
    @inbounds A[I] = KernelAbstractions.__dynamic_checkbounds(__ctx__)
end

# with a workgroup size limited by the kernel's stack use: the sign of `A` is whether it
# checks its indices, the magnitude the size of its workgroups
@kernel function stack_bounds_checked!(A, B, @Const(idx))
    I = @index(Global, Linear)
    tmp = @private Float32 (1000,)
    for j in 1:1000
        @inbounds tmp[j] = j
    end
    @inbounds B[I] = tmp[idx[I]]
    items = prod(@groupsize())
    @inbounds A[I] = KernelAbstractions.__dynamic_checkbounds(__ctx__) ? -items : items
end

@testset "POCL bounds checks" begin
    function checked(kernel, shape; workgroupsize = nothing)
        A = fill(-1, shape)
        kernel(A; ndrange = shape, workgroupsize)
        all(==(A[1]), A) || error("inconsistent bounds checks")
        return Bool(A[1])
    end

    # a kernel without the bounds check runs launches without partial workgroups
    @test !checked(bounds_checked!(CPU()), 64; workgroupsize = 16)
    @test checked(bounds_checked!(CPU()), 60; workgroupsize = 16)
    @test !checked(bounds_checked!(CPU()), (8, 6); workgroupsize = (4, 3))
    @test checked(bounds_checked!(CPU()), (8, 6); workgroupsize = (4, 4))
    @test !checked(bounds_checked!(CPU()), (4, 6, 8); workgroupsize = (2, 3, 4))
    @test checked(bounds_checked!(CPU()), (4, 6, 9); workgroupsize = (2, 3, 4))
    @test !checked(bounds_checked!(CPU(), 8), 16)
    @test checked(bounds_checked!(CPU(), 8), 12)
    # also with a tuned workgroup size, which is a power of two on PoCL, unless a single
    # workgroup covers the ndrange (e.g. on a single thread). an odd ndrange larger than
    # the largest workgroup needs partial workgroups, however many threads PoCL uses.
    @test !checked(bounds_checked!(CPU()), 1024)
    @test checked(bounds_checked!(CPU()), POCL.device_limits().max_work_group_size + 1)
    @test !checked(bounds_checked!(CPU()), (64, 32))

    # only the kernel that the launch needs is compiled
    compilations = POCL.compilations[]
    @test !checked(tuned_bounds_checked!(CPU()), 4096)
    @test POCL.compilations[] == compilations + 1
    @test checked(partial_bounds_checked!(CPU()), 4097)
    @test POCL.compilations[] == compilations + 2

    # the tuned workgroup size is predicted from the device's limits, which a kernel's can
    # be smaller than: then, its workgroups may not cover the ndrange, and need the check
    A, B, idx = zeros(Int, 4096), zeros(Float32, 4096), rand(1:1000, 4096)
    stack_bounds_checked!(CPU())(A, B, idx; ndrange = length(A))
    @test all(==(A[1]), A)
    @test (A[1] < 0) == !iszero(length(A) % abs(A[1]))
    @test B == idx
end

@kernel function muladd_kernel!(D, @Const(A), @Const(B), @Const(C))
    i = @index(Global)
    @inbounds D[i] = muladd(A[i], B[i], C[i])
end

@testset "POCL muladd" begin
    for T in (Float32, Float64)
        A, B, C = rand(T, 64), rand(T, 64), rand(T, 64)
        D = similar(A)
        muladd_kernel!(CPU())(D, A, B, C; ndrange = length(D))
        @test D ≈ muladd.(A, B, C)

        # on CPUs with hardware FMA, the multiplication is not rounded: subtracting the
        # rounded product leaves its rounding error
        a = one(T) + eps(T)
        A, B, C = fill(a, 4), fill(a, 4), fill(-(a * a), 4)
        muladd_kernel!(CPU())(D, A, B, C; ndrange = 4)
        @test all(==(POCL.fast_fma() ? fma(a, a, -(a * a)) : zero(T)), D[1:4])
    end
end

@testset "POCL workgroup size" begin
    workgroupsize = POCL.POCLKernels.cpu_workgroupsize
    for threads in (1, 3, 4, 7, 64), max_items in (1, 7, 64, 4096), nitems in (nothing, 0, 1, 1000, 4096, 2^20, typemax(Int))
        items = workgroupsize(nitems, threads, max_items, 8)
        @test 1 <= items <= max_items
        # a power of two, unless limited by `max_items`
        ispow2(max_items) && @test ispow2(items)
        # splitting stops with more than half as many workgroups as threads, or at the
        # minimum of 4 times the preferred multiple
        if nitems !== nothing
            @test 2 * cld(max(nitems, 1), items) > threads || items < 2 * min(32, max_items)
        end
    end
    # PoCL's choice for the same launches (`pocl_default_local_size_optimizer`)
    for (nitems, threads, items) in (
            (1024, 1, 1024), (1024, 4, 256), (1024, 8, 128), (4096, 4, 1024), (4096, 8, 512),
            (16384, 4, 4096), (16384, 8, 2048), (65536, 8, 4096), (4096, 64, 64),
        )
        @test workgroupsize(nitems, threads, 4096, 8) == items
    end
    # the floor follows the preferred multiple
    @test workgroupsize(1024, 32, 4096, 8) == 32
    @test workgroupsize(1024, 32, 4096, 16) == 64
    # unlike PoCL, launches too small to split over all threads split down to the floor
    @test workgroupsize(1000, 64, 4096, 8) == 32
    @test workgroupsize(4096, 192, 4096, 8) == 32
    @test workgroupsize(64, 64, 4096, 8) == 32
    # more threads don't mean larger workgroups
    @test issorted([workgroupsize(4096, threads, 4096, 8) for threads in 1:64]; rev = true)
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

        # the launch is chosen from the iteration space, computing indices in `Int` on PoCL
        @test KA.select_launch(kernel, nothing, iterspace) === KA.NDLaunch{Int}()
    end
    for launch in (nothing, KA.LinearLaunch{Int}(), KA.NDLaunch{Int}())
        function launcher(kernel, args...; ndrange, workgroupsize = nothing)
            ndrange, workgroupsize, iterspace, _ = KA.launch_config(kernel, ndrange, workgroupsize)
            # an N-d launch needs one work-item past the third dimension, as tuned workgroups have
            fold = KA.workgroupsize(kernel) <: KA.DynamicSize && workgroupsize === nothing ||
                all(==(1), size(KA.workitems(iterspace))[4:end])
            l = launch isa KA.NDLaunch && !fold ? KA.LinearLaunch{Int}() : launch
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

# a GPU back-end, whose kernel type depends on the function and argument types
struct StaticBackend <: KernelAbstractions.GPU end
struct StaticKernel{F, TT} end
const KI = KernelAbstractions.KernelInterface
KI.argconvert(::StaticBackend, x) = x
KI.kernel_function(b::StaticBackend, ::F, ::Type{TT}; kwargs...) where {F, TT} =
    KI.Kernel{StaticBackend, StaticKernel{F, TT}}(b, StaticKernel{F, TT}())
KI.launch(::KI.Kernel{StaticBackend}, groups, items, args) = nothing
KI.max_work_group_size(::Union{StaticBackend, KI.Kernel{StaticBackend}}) = 1024
KI.max_work_group_dims(::StaticBackend) = (1024, 1024, 64)
KI.max_num_groups(::StaticBackend) = (2^31 - 1, 65535, 65535)
KI.launch_configuration(::KI.Kernel{StaticBackend}; kwargs...) = (; workgroupsize = 256)
@testset "Kernel type is inferred" begin
    KA = KernelAbstractions
    kernel = mapped_indices!(StaticBackend())
    @testset "workgroupsize $ws" for ws in (nothing, 8)
        ndrange, _, iterspace, _ = KA.launch_config(kernel, 10, ws)
        tt = Tuple{
            typeof(kernel), Nothing, Nothing, typeof(ndrange), typeof(ws), typeof(iterspace),
            Tuple{Vector{Int}}, Bool,
        }
        ci, _ = only(code_typed(KA.launch_kernel, tt; optimize = false))
        kernels = filter(T -> T !== Union{} && T <: KI.Kernel, map(Core.Compiler.widenconst, ci.ssavaluetypes))
        @test !isempty(kernels)
        @test all(isconcretetype, kernels)
    end
end

@testset "Deprecated GPU alias" begin
    @test KernelAbstractions.GPU === KernelAbstractions.Backend
    @test CPU() isa KernelAbstractions.GPU
    @test NewBackend <: KernelAbstractions.GPU
end
