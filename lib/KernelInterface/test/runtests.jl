# These are the standalone tests for KernelInterface

using KernelInterface
using Aqua
using Test

const KI = KernelInterface

# `_print`'s host fallback writes to `stdout`, so capture it through a real file.
function capture_stdout(f)
    return mktemp() do path, io
        redirect_stdout(f, io)
        flush(io)
        return read(path, String)
    end
end

@testset "standalone" begin
    # KernelInterface is what backends implement against, so it must stay loadable
    # without dragging in KernelAbstractions or a compiler stack.
    toml = read(joinpath(pkgdir(KernelInterface), "Project.toml"), String)
    @test !occursin("[deps]", toml)
    @test !occursin("[sources]", toml)
end

struct ShuffleBackend <: KI.Backend end
KI.supports_shuffle(::ShuffleBackend, ::Type{Int32}) = true

# NOTE: this runs before the mock backend below defines methods on `argconvert`
# and `kernel_function`.
@testset "interface stubs" begin
    # These have no fallback on purpose: a backend that forgets to `@device_override`
    # them should get a MethodError rather than silently wrong behaviour.
    stubs = [
        KI.sub_group_any, KI.sub_group_all, KI.sub_group_ballot,
        KI.max_work_group_size, KI.max_work_group_dims, KI.max_num_groups,
        KI.sub_group_size, KI.argconvert, KI.kernel_function, KI.launch,
        # Host-side stubs: required backend methods with no sensible fallback.
        KI.synchronize, KI.copyto!,
    ]
    for stub in stubs
        @test isempty(methods(stub))
    end

    # The shuffles only have the fallback that shuffles structs field by field, which
    # doesn't handle the primitive types a backend has to implement.
    for shfl in [KI.shfl, KI.shfl_down, KI.shfl_up, KI.shfl_xor]
        @test_throws ArgumentError shfl(1.0f0, 1)
        @test_throws ArgumentError shfl((1.0f0, 2), 1)
        @test_throws ArgumentError shfl(Ref(1), 1)
    end
    @test !KI.supports_shuffle(ShuffleBackend(), Float32)
    @test !KI.supports_shuffle(ShuffleBackend(), Tuple{Float32, Int})
    @test KI.supports_shuffle(ShuffleBackend(), Int32)
    @test KI.supports_shuffle(ShuffleBackend(), Tuple{Int32, NTuple{2, Int32}})
    @test !KI.supports_shuffle(ShuffleBackend(), Tuple{Int32, Float32})

    # The primitive queries take an element type; only the zero-argument form has a
    # (forwarding) method, and it must reach the typed stub rather than recurse.
    primitives = [
        KI.get_local_size, KI.get_local_id,
        KI.get_num_groups, KI.get_group_id,
        KI.get_sub_group_size, KI.get_max_sub_group_size,
        KI.get_num_sub_groups, KI.get_sub_group_id,
        KI.get_sub_group_local_id,
    ]
    for f in primitives
        @test length(methods(f)) == 1
        @test hasmethod(f, Tuple{})
        @test !hasmethod(f, Tuple{Type{Int}})
        @test_throws MethodError f()
        @test_throws MethodError f(Int32)
    end

    # The global queries are derived from the primitive ones.
    for f in [KI.get_global_size, KI.get_global_id]
        @test hasmethod(f, Tuple{Type{Int}})
        @test_throws MethodError f()
        @test_throws MethodError f(Int32)
    end
end

struct StubBackend <: KI.Backend end

# A backend with two devices that forgot the other device functions.
struct MultiDeviceBackend <: KI.Backend end
KI.ndevices(::MultiDeviceBackend) = 2

# An array type with a known backend, for exercising the `get_backend` fallback
# that unwraps wrapper arrays.
struct BackedArray{T, N} <: AbstractArray{T, N}
    data::Array{T, N}
end
Base.size(A::BackedArray) = size(A.data)
Base.getindex(A::BackedArray{T, N}, i::Vararg{Int, N}) where {T, N} = A.data[i...]
KI.get_backend(::BackedArray) = StubBackend()

# A backend implementing only `allocate`, as the interface requires.
struct AllocBackend <: KI.Backend end
function KI.allocate(::AllocBackend, ::Type{T}, dims::Tuple; unified::Bool = false) where {T}
    return Array{T}(undef, dims)
end

@testset "host fallbacks" begin
    # Barriers are meaningless off-device and must say so rather than no-op.
    @test_throws "used outside kernel" KI.barrier()
    @test_throws "used outside kernel" KI.sub_group_barrier()

    # Conservative defaults: a backend only implements these if it can do better.
    @test KI.multiprocessor_count(StubBackend()) == 0
    @test KI.supports_subgroups(StubBackend()) === false
    @test KI.supports_shuffle(StubBackend(), Float32) === false

    # `localmemory` forwards the untyped `dims` to the `Val` form backends override.
    # Off-device that form is unimplemented, and must error rather than recurse
    # back into the forwarding method.
    @test_throws "used outside kernel" KI.localmemory(Float32, (2, 2))
    @test_throws "used outside kernel" KI.localmemory(Float32, Val((2, 2)))
end

@testset "get_backend" begin
    # The fallback finds the backend of wrapper arrays by walking `parent`.
    arr = BackedArray([1, 2, 3])
    @test KI.get_backend(arr) === StubBackend()
    @test KI.get_backend(view(arr, 1:2)) === StubBackend()
    @test KI.get_backend(reshape(arr, 3, 1)) === StubBackend()
    @test KI.get_backend(reinterpret(UInt, arr)) === StubBackend()

    # An array that is its own parent has no wrapped backend to find; the
    # fallback must error rather than recurse.
    @test_throws ArgumentError KI.get_backend([1, 2, 3])
end

@testset "backend queries" begin
    b = StubBackend()

    # `versioninfo` falls back to printing a notice, defaulting to `stdout`.
    @test occursin("not implemented", sprint(KI.versioninfo, b))
    @test occursin("not implemented", capture_stdout(() -> KI.versioninfo(b)))

    # `missing` distinguishes "not implemented" from a definite yes/no.
    @test KI.functional(b) === missing

    # Single-device defaults; `device!` still bounds-checks the id.
    @test KI.device(b) == 1
    @test KI.ndevices(b) == 1
    @test KI.device!(b, 1) === nothing
    @test KI.device(b, zeros(2)) == 1
    @test_throws ArgumentError KI.device!(b, 0)
    @test_throws ArgumentError KI.device!(b, 2)

    # A backend with several devices that only implements `ndevices` gets errors from
    # the single-device fallbacks, not answers for the wrong device.
    mb = MultiDeviceBackend()
    @test_throws "must implement `KernelInterface.device`" KI.device(mb)
    @test_throws "must implement `KernelInterface.device`" KI.device(mb, zeros(2))
    @test_throws "must implement `KernelInterface.device!`" KI.device!(mb, 2)
    @test_throws ArgumentError KI.device!(mb, 3)

    # `priority!` validates the symbol even when the backend ignores it.
    for prio in (:high, :normal, :low)
        @test KI.priority!(b, prio) === nothing
    end
    @test_throws "priority must be one of" KI.priority!(b, :bogus)

    # Capability defaults are conservative: a missing method never claims support.
    @test KI.supports_unified(b) === false
    @test KI.supports_atomics(b) === false
    @test KI.supports_float64(b) === false

    # Pinning is optional and freeing is a no-op unless a backend does better.
    @test KI.pagelock!(b, zeros(2)) === missing
    @test KI.unsafe_free!(zeros(2)) === nothing
end

# A backend implementing only `synchronize`, for exercising the event fallbacks.
struct SyncBackend <: KI.Backend
    synchronizations::Base.RefValue{Int}
end
SyncBackend() = SyncBackend(Ref(0))
KI.synchronize(b::SyncBackend) = (b.synchronizations[] += 1; nothing)

@testset "record_event / wait_event" begin
    b = SyncBackend()

    # Without an event type of its own, a backend records by synchronizing fully, and
    # the resulting `nothing` handle is a no-op to wait on.
    @test KI.record_event(b) === nothing
    @test b.synchronizations[] == 1
    @test KI.wait_event(b, nothing) === nothing
    @test b.synchronizations[] == 1

    # A backend that does not implement `synchronize` cannot record either.
    @test_throws MethodError KI.record_event(StubBackend())
    # Only events a backend defines `wait_event` for are accepted.
    @test_throws MethodError KI.wait_event(b, :bogus)
end

@testset "allocate / zeros / ones" begin
    b = AllocBackend()

    # Dims given as varargs are forwarded to the tuple method backends implement.
    @test KI.allocate(b, Float32, (2,)) isa Vector{Float32}
    @test size(KI.allocate(b, Float32, 2, 3)) == (2, 3)

    @test KI.zeros(b, Float64, 2, 3) == zeros(2, 3)
    @test KI.ones(b, Int, (4,)) == ones(Int, 4)

    # A backend without `allocate` yields a MethodError pointing at the missing
    # method — including via the keyword form — and a clear error when unified
    # memory is requested but not supported.
    @test_throws MethodError KI.allocate(StubBackend(), Float32, (2,))
    @test_throws MethodError KI.allocate(StubBackend(), Float32, (2,); unified = false)
    @test_throws ArgumentError KI.allocate(StubBackend(), Float32, (2,); unified = true)
end

@testset "_print" begin
    # The host fallback keeps `KernelAbstractions.@print` working outside a kernel.
    # `@print` wraps literals in `Val` so backends can use them as format strings;
    # the fallback has to unwrap them again.
    @test capture_stdout(() -> KI._print()) == ""
    @test capture_stdout(() -> KI._print(Val(Symbol("hello\n")))) == "hello\n"
    @test capture_stdout(() -> KI._print(1, 2)) == "12"
    @test capture_stdout(() -> KI._print(Val(Symbol("x = ")), 42, Val(Symbol("\n")))) ==
        "x = 42\n"
    @test capture_stdout(() -> KI._print(Val(3), " ", Val(:sym))) == "3 sym"
end

@testset "threads_to_workgroupsize" begin
    # Fills dimensions left to right without exceeding the thread budget.
    @test KI.threads_to_workgroupsize(256, (1000,)) == (256,)
    @test KI.threads_to_workgroupsize(256, (100,)) == (100,)
    @test KI.threads_to_workgroupsize(256, (100, 50)) == (100, 2)
    @test KI.threads_to_workgroupsize(1024, (5, 5, 5)) == (5, 5, 5)
    @test KI.threads_to_workgroupsize(4, (3, 3)) == (3, 1)
    @test prod(KI.threads_to_workgroupsize(256, (100, 50))) <= 256

    # Zero-sized dimensions are clamped to 1 so the launch math stays defined.
    @test KI.threads_to_workgroupsize(256, (0, 4)) == (1, 4)
    @test KI.threads_to_workgroupsize(256, (4, 0)) == (4, 1)
    @test KI.threads_to_workgroupsize(0, (5,)) == (1,)

    # Per-dimension limits, as for CUDA's (1024, 1024, 64) blocks.
    @test KI.threads_to_workgroupsize(1024, (1, 1, 5000), (1024, 1024, 64)) == (1, 1, 64)
    @test KI.threads_to_workgroupsize(1024, (2000, 3), (512, 1024, 64)) == (512, 2)
    # dimensions past the limits are only bounded by the thread budget
    @test KI.threads_to_workgroupsize(64, (1, 1, 1, 100), (1024, 1024, 64)) == (1, 1, 1, 64)
end

@testset "Kernel" begin
    kernel = KI.Kernel(:backend, :kern)
    @test kernel.backend === :backend
    @test kernel.kern === :kern
end

# A minimal backend, recording the compilations and launches that KernelInterface asks for.
struct MockBackend <: KI.Backend
    max_items::Int
end
MockBackend() = MockBackend(256)

struct MockKernel
    f::Any
    tt::Any
    name::Any
    options::Any
    launches::Vector{Any}
end

KI.argconvert(::MockBackend, arg) = arg
function KI.kernel_function(backend::MockBackend, f, tt = Tuple{}; name = nothing, kwargs...)
    return KI.Kernel(backend, MockKernel(f, tt, name, Dict(kwargs), []))
end
function KI.launch(kernel::KI.Kernel{MockBackend}, groups::Dims{3}, items::Dims{3}, args::Tuple; kwargs...)
    push!(kernel.kern.launches, (; groups, items, args, kwargs = Dict(kwargs)))
    return :ignored
end
KI.max_work_group_size(kernel::KI.Kernel{MockBackend}) = kernel.backend.max_items
KI.max_work_group_dims(::MockBackend) = (1024, 1024, 64)

# ... and one recommending smaller work-groups than it can launch, like CUDA's occupancy API,
# recording what it was asked
struct OccupancyBackend <: KI.Backend
    queries::Vector{Any}
end
OccupancyBackend() = OccupancyBackend([])
KI.max_work_group_size(::KI.Kernel{OccupancyBackend}) = 1024
KI.max_work_group_dims(::OccupancyBackend) = (1024, 1024, 64)
function KI.launch_configuration(
        kernel::KI.Kernel{OccupancyBackend}; nitems = nothing, max_work_group_size = typemax(Int)
    )
    push!(kernel.backend.queries, (; nitems, max_work_group_size))
    return (; workgroupsize = min(96, max_work_group_size))
end
KI.launch(kernel::KI.Kernel{OccupancyBackend}, groups::Dims{3}, items::Dims{3}, args::Tuple) =
    push!(kernel.kern, (groups, items))

# a callable that KernelInterface mustn't convert: the backend does
struct HostCallable end
(::HostCallable)(x) = nothing
KI.argconvert(::MockBackend, ::HostCallable) = error("only the backend should convert the callable")

# ... and one that does nothing, to measure the overhead of launching
struct NullBackend <: KI.Backend end
KI.max_work_group_size(::KI.Kernel{NullBackend}) = 1024
KI.max_work_group_dims(::NullBackend) = (1024, 1024, 64)
KI.launch(::KI.Kernel{NullBackend}, groups::Dims{3}, items::Dims{3}, args::Tuple; kwargs...) = nothing

@testset "launch geometry" begin
    kernel = KI.kernel_function(MockBackend(), identity, Tuple{Int})
    function geometry(; kwargs...)
        empty!(kernel.kern.launches)
        @test kernel(1; kwargs...) === nothing
        isempty(kernel.kern.launches) && return nothing
        launch = only(kernel.kern.launches)
        return launch.groups, launch.items
    end

    # Without an ndrange the sizes pass through, defaulting to 1.
    @test geometry() == ((1, 1, 1), (1, 1, 1))
    @test geometry(numgroups = 4) == ((4, 1, 1), (1, 1, 1))
    @test geometry(workgroupsize = (2, 2)) == ((1, 1, 1), (2, 2, 1))
    @test geometry(numgroups = (4, 3), workgroupsize = (2, 5)) == ((4, 3, 1), (2, 5, 1))
    @test geometry(numgroups = (4, 3, 2), workgroupsize = (2, 5, 3)) == ((4, 3, 2), (2, 5, 3))

    # With an ndrange and no workgroupsize, the workgroupsize is derived from
    # the kernel's limit and the workgroup count covers the ndrange.
    @test geometry(ndrange = 1000) == ((4, 1, 1), (256, 1, 1))
    @test geometry(ndrange = (1000,)) == ((4, 1, 1), (256, 1, 1))
    @test geometry(ndrange = 10) == ((1, 1, 1), (10, 1, 1))
    @test geometry(ndrange = (100, 50)) == ((1, 25, 1), (100, 2, 1))
    @test geometry(ndrange = 1000, max_work_group_size = 100) == ((10, 1, 1), (100, 1, 1))
    # ... also for ndranges with more elements than an `Int` can count
    @test geometry(ndrange = (2^40, 2^40)) == ((2^32, 2^40, 1), (256, 1, 1))
    # ... respecting the per-dimension limit
    let k = KI.kernel_function(MockBackend(1024), identity)
        k(; ndrange = (1, 1, 5000))
        launch = only(k.kern.launches)
        @test (launch.groups, launch.items) == ((1, 1, 79), (1, 1, 64))
    end

    # An explicit workgroupsize is kept as-is, and the ndrange rounded up to it.
    @test geometry(ndrange = 100, workgroupsize = 16) == ((7, 1, 1), (16, 1, 1))
    @test geometry(ndrange = (7, 5), workgroupsize = (2, 3)) == ((4, 2, 1), (2, 3, 1))

    # Zero anywhere in the ndrange or the number of groups launches nothing.
    @test geometry(ndrange = 0) === nothing
    @test geometry(ndrange = (0, 4)) === nothing
    @test geometry(ndrange = (4, 0), workgroupsize = 2) === nothing
    @test geometry(numgroups = (2, 0)) === nothing
    @test geometry(ndrange = (0, typemax(Int)), workgroupsize = (1, 2)) === nothing

    # Invalid launches are rejected before the backend sees them.
    for kwargs in [
            (; numgroups = (1, 1, 1, 1)), (; workgroupsize = (1, 1, 1, 1)),
            (; ndrange = (1, 1, 1, 1)), (; ndrange = 2, numgroups = 2),
            (; workgroupsize = 0), (; workgroupsize = (1, 0)), (; workgroupsize = -1),
            (; numgroups = -1), (; ndrange = -1), (; ndrange = 2.0), (; numgroups = [1]),
            (; ndrange = 4, max_work_group_size = 0), (; ndrange = typemax(UInt)),
            # the kernel's limit, and the per-dimension limit
            (; workgroupsize = 257), (; workgroupsize = (1, 1, 65)),
            # more work-items in a dimension than an `Int` can count
            (; ndrange = typemax(Int), workgroupsize = 2),
            (; numgroups = typemax(Int), workgroupsize = 2),
            (; numgroups = (1, typemax(Int) ÷ 2 + 1), workgroupsize = (1, 2)),
        ]
        @test_throws ArgumentError kernel(1; kwargs...)
    end
    @test isempty(kernel.kern.launches)

    # Other keywords are for the backend.
    kernel(1; ndrange = 4, stream = :mine)
    @test last(kernel.kern.launches).kwargs == Dict(:stream => :mine)

    # The arguments reach the backend as one tuple, whatever their number, and a single
    # tuple-valued argument stays one argument.
    kernel((1, 2); ndrange = 4)
    @test last(kernel.kern.launches).args == ((1, 2),)
    kernel(ntuple(identity, 40)...; ndrange = 4)
    @test last(kernel.kern.launches).args == ntuple(identity, 40)

    # Auto-sizing uses the backend's recommendation, not the limit, and tells it both the
    # size of the launch and the cap.
    occupancy = KI.Kernel(OccupancyBackend(), [])
    occupancy(; ndrange = 1000)
    @test only(occupancy.kern) == ((11, 1, 1), (96, 1, 1))
    @test only(occupancy.backend.queries) == (; nitems = 1000, max_work_group_size = typemax(Int))
    occupancy(; ndrange = (1000, 3), max_work_group_size = 64)
    @test last(occupancy.kern) == ((16, 3, 1), (64, 1, 1))
    @test last(occupancy.backend.queries) == (; nitems = 3000, max_work_group_size = 64)
    occupancy(; ndrange = (2^40, 2^40))
    @test last(occupancy.backend.queries).nitems == typemax(Int)
    # ... while explicit sizes can go up to the limit
    occupancy(; workgroupsize = 1024)
    @test last(occupancy.kern) == ((1, 1, 1), (1024, 1, 1))
end

@testset "launch_configuration" begin
    kernel = KI.kernel_function(MockBackend(), identity)
    # the fallback recommends the limit
    @test KI.launch_configuration(kernel) === (; workgroupsize = 256)
    @test KI.launch_configuration(kernel; max_work_group_size = 100) === (; workgroupsize = 100)
    @test KI.launch_configuration(kernel; nitems = 10) === (; workgroupsize = 256)

    # backends can recommend less than the limit
    occupancy = KI.Kernel(OccupancyBackend(), [])
    @test KI.launch_configuration(occupancy) === (; workgroupsize = 96)
    @test KI.max_work_group_size(occupancy) == 1024
end

@testset "split_kwargs" begin
    kwargs = [:(launch = false), :(name = "foo"), :(numgroups = 2)]
    macro_kw, launch_kw, other = KI.split_kwargs(kwargs, KI.MACRO_KWARGS, KI.LAUNCH_KWARGS)
    @test macro_kw == [:(launch = false)]
    @test launch_kw == [:(numgroups = 2)]
    @test other == [:(name = "foo")]

    # Unmatched keywords land in the trailing group rather than erroring.
    _, unmatched = KI.split_kwargs([:(bogus = 1)], [:launch])
    @test unmatched == [:(bogus = 1)]

    # Also usable at run time with pairs instead of expressions.
    matched, _ = KI.split_kwargs([:launch => false], [:launch])
    @test matched == [:launch => false]

    @test_throws ArgumentError KI.split_kwargs([:(f(x))], [:launch])
    @test_throws ArgumentError KI.split_kwargs([Expr(:(=), 1, 2)], [:launch])
end

@testset "assign_args!" begin
    code = Expr(:block)
    vars, var_exprs = KI.assign_args!(code, [:a, :(b...)])
    @test length(vars) == 2
    # Arguments are hoisted into gensyms so the caller can `GC.@preserve` them.
    @test code.args == [:($(vars[1]) = a), :($(vars[2]) = b)]
    @test var_exprs[1] === vars[1]
    @test var_exprs[2] == Expr(:..., vars[2])
end

dummy(a, b) = nothing

const backend_evaluations = Ref(0)
function counted_backend()
    backend_evaluations[] += 1
    return MockBackend()
end

# Julia doesn't turn a splat of more than 32 elements into a direct call, so launching with
# many arguments allocates unless they're passed on as a tuple
@testset "many arguments" begin
    kernel = KI.Kernel(NullBackend(), nothing)
    @eval launch_few(k) = k($((1:4)...); numgroups = 2, workgroupsize = 4)
    @eval launch_many(k) = k($((1:40)...); numgroups = 2, workgroupsize = 4)
    launch_few(kernel)
    launch_many(kernel)
    @test @allocated(launch_many(kernel)) <= @allocated(launch_few(kernel))
end

@testset "@launch" begin
    backend = MockBackend()

    kernel = KI.@launch backend numgroups = 2 workgroupsize = 4 dummy(1, 2.0)
    @test kernel isa KI.Kernel{MockBackend}
    @test kernel.backend === backend
    @test kernel.kern.f === dummy
    @test kernel.kern.tt == Tuple{Int, Float64}
    launch = only(kernel.kern.launches)
    @test launch.args == (1, 2.0)
    @test (launch.groups, launch.items) == ((2, 1, 1), (4, 1, 1))

    # the backend expression is evaluated once
    backend_evaluations[] = 0
    KI.@launch counted_backend() ndrange = 4 dummy(1, 2.0)
    @test backend_evaluations[] == 1

    # `launch=false` compiles only; the caller launches later.
    deferred = KI.@launch backend launch = false dummy(1, 2.0)
    @test isempty(deferred.kern.launches)

    # Other keywords are compiler options for `kernel_function`.
    named = KI.@launch backend launch = false name = "mykernel" maxthreads = 32 dummy(1, 2.0)
    @test named.kern.name == "mykernel"
    @test named.kern.options == Dict(:maxthreads => 32)
    optioned = KI.@launch backend ndrange = 4 maxthreads = 32 dummy(1, 2.0)
    @test isempty(only(optioned.kern.launches).kwargs)

    # The callable is compiled unconverted.
    @test (KI.@launch backend launch = false HostCallable()(1)).kern.f isa HostCallable

    # Splatted arguments are supported.
    splatted = KI.@launch backend launch = false dummy((1, 2.0)...)
    @test splatted.kern.tt == Tuple{Int, Float64}

    @testset "errors" begin
        # These throw during macro expansion, so they cannot be written as a plain
        # `@test_throws` call. `macroexpand` wraps such errors in a `LoadError`.
        function expansion_error(ex)
            try
                macroexpand(@__MODULE__, ex)
            catch err
                return err isa LoadError ? err.error : err
            end
            return nothing
        end

        @test expansion_error(:(KI.@launch backend)) isa ArgumentError
        @test expansion_error(:(KI.@launch backend dummy)) isa ArgumentError
        @test expansion_error(:(KI.@launch backend launch = 1 dummy(1))) isa ArgumentError
        @test expansion_error(:(KI.@launch backend "notakwarg" dummy(1))) isa ArgumentError
        # launch keywords are meaningless when we are not launching
        @test expansion_error(
            :(KI.@launch backend launch = false numgroups = 2 dummy(1))
        ) isa ArgumentError
    end
end

@testset "Aqua" begin
    Aqua.test_all(KernelInterface)
end
