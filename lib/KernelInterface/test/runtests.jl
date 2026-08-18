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

# NOTE: this runs before the mock backend below defines methods on `argconvert`
# and `kernel_function`.
@testset "interface stubs" begin
    # These have no fallback on purpose: a backend that forgets to `@device_override`
    # them should get a MethodError rather than silently wrong behaviour.
    stubs = [
        KI.get_global_size, KI.get_global_id,
        KI.get_local_size, KI.get_local_id,
        KI.get_num_groups, KI.get_group_id,
        KI.get_sub_group_size, KI.get_max_sub_group_size,
        KI.get_num_sub_groups, KI.get_sub_group_id,
        KI.get_sub_group_local_id,
        KI.shfl_down,
        KI.kernel_max_work_group_size, KI.max_work_group_size, KI.sub_group_size,
        KI.argconvert, KI.kernel_function,
        # Host-side stubs: required backend methods with no sensible fallback.
        KI.synchronize, KI.copyto!,
    ]
    for stub in stubs
        @test isempty(methods(stub))
    end
end

struct StubBackend <: KI.Backend end

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

    # Permissive defaults: a backend only implements these if it can do better.
    @test KI.shfl_down_types(StubBackend()) == DataType[]
    @test KI.multiprocessor_count(StubBackend()) == 0

    # `localmemory` forwards the untyped `dims` to the `Val` form backends override.
    # Off-device that form is unimplemented, and must error rather than recurse
    # back into the forwarding method.
    @test_throws "used outside kernel" KI.localmemory(Float32, (2, 2))
    @test_throws "used outside kernel" KI.localmemory(Float32, Val((2, 2)))
end

@testset "get_backend" begin
    @test KI.GPU <: KI.Backend

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
    @test_throws ArgumentError KI.device!(b, 0)
    @test_throws ArgumentError KI.device!(b, 2)

    # `priority!` validates the symbol even when the backend ignores it.
    for prio in (:high, :normal, :low)
        @test KI.priority!(b, prio) === nothing
    end
    @test_throws "priority must be one of" KI.priority!(b, :bogus)

    # Capability defaults: pessimistic for unified memory, optimistic otherwise.
    @test KI.supports_unified(b) === false
    @test KI.supports_atomics(b) === true
    @test KI.supports_float64(b) === true

    # Pinning is optional and freeing is a no-op unless a backend does better.
    @test KI.pagelock!(b, zeros(2)) === missing
    @test KI.unsafe_free!(zeros(2)) === nothing
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

@testset "check_launch_args" begin
    # Validation only: valid configurations pass through without normalization.
    @test KI.check_launch_args(1, 1, ()) === nothing
    @test KI.check_launch_args((1, 2, 3), (1, 2, 3), ()) === nothing
    @test KI.check_launch_args((), 1, 1) === nothing
    @test KI.check_launch_args((), (), ()) === nothing

    @test_throws ArgumentError KI.check_launch_args((1, 2, 3, 4), 1, ())
    @test_throws ArgumentError KI.check_launch_args(1, (1, 2, 3, 4), ())
    @test_throws ArgumentError KI.check_launch_args((), 1, (1, 2, 3, 4))
    @test_throws ArgumentError KI.check_launch_args(2, 4, 2) # both numworkgroupsize and ndrange defined
    @test_throws ArgumentError KI.check_launch_args(2, (), 2) # both numworkgroupsize and ndrange defined
end

@testset "threads_to_workgroupsize" begin
    # Fills dimensions left to right without exceeding the thread budget.
    @test KI.threads_to_workgroupsize(256, (1000,)) == (256,)
    @test KI.threads_to_workgroupsize(256, (100,)) == (100,)
    @test KI.threads_to_workgroupsize(256, (100, 50)) == (100, 2)
    @test KI.threads_to_workgroupsize(1024, (5, 5, 5)) == (5, 5, 5)
    @test KI.threads_to_workgroupsize(4, (3, 3)) == (3, 1)
    @test prod(KI.threads_to_workgroupsize(256, (100, 50))) <= 256
end

@testset "Kernel" begin
    kernel = KI.Kernel(:backend, :kern)
    @test kernel.backend === :backend
    @test kernel.kern === :kern
end

# A backend reporting a fixed workgroup-size limit, for exercising the
# auto-sizing helper.
struct SizedBackend <: KI.Backend
    maxThreads::Int
end
function KI.kernel_max_work_group_size(k::KI.Kernel{SizedBackend}; max_work_items::Int = typemax(Int))
    return min(k.backend.maxThreads, max_work_items)
end

@testset "auto_launch_sizes" begin
    kernel = KI.Kernel(SizedBackend(256), nothing)

    # Without an ndrange the sizes pass through, defaulting to 1.
    @test KI.auto_launch_sizes(kernel, (), (), ()) === (1, 1)
    @test KI.auto_launch_sizes(kernel, 4, (), ()) === (4, 1)
    @test KI.auto_launch_sizes(kernel, (), (2, 2), ()) === (1, (2, 2))
    @test KI.auto_launch_sizes(kernel, (4, 4), (2, 2), ()) === ((4, 4), (2, 2))

    # With an ndrange and no workgroupsize, the workgroupsize is derived from
    # the kernel's limit and the workgroup count covers the ndrange.
    @test KI.auto_launch_sizes(kernel, (), (), (1000,)) === ((4,), (256,))
    @test KI.auto_launch_sizes(kernel, (), (), (10,)) === ((1,), (10,))
    @test KI.auto_launch_sizes(kernel, (), (), (100, 50)) === ((1, 25), (100, 2))
    @test KI.auto_launch_sizes(kernel, (), (), 1000) === (4, 256)

    # An explicit workgroupsize is kept as-is.
    @test KI.auto_launch_sizes(kernel, (), (16,), (100,)) === ((7,), (16,))
end

@testset "split_kwargs" begin
    kwargs = [:(launch = false), :(name = "foo"), :(numworkgroups = 2)]
    macro_kw, compiler_kw, launch_kw, other = KI.split_kwargs(
        kwargs, KI.MACRO_KWARGS, KI.COMPILER_KWARGS, KI.LAUNCH_KWARGS
    )
    @test macro_kw == [:(launch = false)]
    @test compiler_kw == [:(name = "foo")]
    @test launch_kw == [:(numworkgroups = 2)]
    @test isempty(other)

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

# A minimal backend, exercising the contract `KI.@kernel` expects of one.
struct MockBackend end

struct MockKernel
    f::Any
    tt::Any
    name::Any
    launches::Vector{Any}
end

KI.argconvert(::MockBackend, arg) = arg
function KI.kernel_function(::MockBackend, f, tt = Tuple{}; name = nothing, kwargs...)
    return MockKernel(f, tt, name, [])
end
function (kernel::MockKernel)(args...; kwargs...)
    push!(kernel.launches, (args, Dict(kwargs)))
    return nothing
end

dummy(a, b) = nothing

@testset "@kernel" begin
    backend = MockBackend()

    kernel = KI.@kernel backend numworkgroups = 2 workgroupsize = 4 dummy(1, 2.0)
    @test kernel isa MockKernel
    @test kernel.f === dummy
    @test kernel.tt == Tuple{Int, Float64}
    args, launch_kwargs = only(kernel.launches)
    @test args == (1, 2.0)
    @test launch_kwargs == Dict(:numworkgroups => 2, :workgroupsize => 4)

    # `launch=false` compiles only; the caller launches later.
    deferred = KI.@kernel backend launch = false dummy(1, 2.0)
    @test isempty(deferred.launches)

    # Compiler kwargs reach `kernel_function` instead of the launch.
    named = KI.@kernel backend launch = false name = "mykernel" dummy(1, 2.0)
    @test named.name == "mykernel"

    # Splatted arguments are supported.
    splatted = KI.@kernel backend launch = false dummy((1, 2.0)...)
    @test splatted.tt == Tuple{Int, Float64}

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

        @test expansion_error(:(KI.@kernel backend bogus = 1 dummy(1))) isa ArgumentError
        @test expansion_error(:(KI.@kernel backend dummy)) isa ArgumentError
        @test expansion_error(:(KI.@kernel backend launch = 1 dummy(1))) isa ArgumentError
        @test expansion_error(:(KI.@kernel backend "notakwarg" dummy(1))) isa ArgumentError
        # launch-time kwargs are meaningless when we are not launching
        @test expansion_error(
            :(KI.@kernel backend launch = false numworkgroups = 2 dummy(1))
        ) isa ErrorException
    end
end

@testset "Aqua" begin
    Aqua.test_all(KernelInterface)
end
