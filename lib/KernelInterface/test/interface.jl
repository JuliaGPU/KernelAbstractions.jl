import KernelInterface as KI
using Random

# Counts every work-item at the element it identifies, computed from the group and local
# ids, so that a mix-up between group counts and group sizes shows.
function launch_kernel(arr)
    l = KI.get_local_id()
    g = KI.get_group_id()
    s = KI.get_local_size()
    i = (g.x - 1) * s.x + l.x
    j = (g.y - 1) * s.y + l.y
    k = (g.z - 1) * s.z + l.z
    if i <= size(arr, 1) && j <= size(arr, 2) && k <= size(arr, 3)
        @inbounds arr[i, j, k] += 1
    end
    return
end

struct KernelData
    global_size::Int
    global_id::Int
    local_size::Int
    local_id::Int
    num_groups::Int
    group_id::Int
end
function test_interface_kernel(results)
    i = KI.get_global_id().x

    if i <= length(results)
        @inbounds results[i] = KernelData(
            KI.get_global_size().x,
            KI.get_global_id().x,
            KI.get_local_size().x,
            KI.get_local_id().x,
            KI.get_num_groups().x,
            KI.get_group_id().x
        )
    end
    return
end
struct SubgroupData
    sub_group_size::UInt32
    max_sub_group_size::UInt32
    num_sub_groups::UInt32
    sub_group_id::UInt32
    sub_group_local_id::UInt32
end
function test_subgroup_kernel(results)
    i = KI.get_global_id().x

    if i <= length(results)
        @inbounds results[i] = SubgroupData(
            KI.get_sub_group_size(),
            KI.get_max_sub_group_size(),
            KI.get_num_sub_groups(),
            KI.get_sub_group_id(),
            KI.get_sub_group_local_id()
        )
    end
    return
end

# The interface documents a concrete return type for each device-side function;
# these kernels record whether the backend honors them.
const WorkItemNT{T} = @NamedTuple{x::T, y::T, z::T}

function typecheck_kernel(results)
    @inbounds begin
        results[1] = KI.get_global_size() isa WorkItemNT{Int}
        results[2] = KI.get_global_id() isa WorkItemNT{Int}
        results[3] = KI.get_local_size() isa WorkItemNT{Int}
        results[4] = KI.get_local_id() isa WorkItemNT{Int}
        results[5] = KI.get_num_groups() isa WorkItemNT{Int}
        results[6] = KI.get_group_id() isa WorkItemNT{Int}
    end
    return
end

# The indexing queries take an element type; the result must use it.
function typed_typecheck_kernel(results, ::Type{T}) where {T}
    @inbounds begin
        results[1] = KI.get_global_size(T) isa WorkItemNT{T}
        results[2] = KI.get_global_id(T) isa WorkItemNT{T}
        results[3] = KI.get_local_size(T) isa WorkItemNT{T}
        results[4] = KI.get_local_id(T) isa WorkItemNT{T}
        results[5] = KI.get_num_groups(T) isa WorkItemNT{T}
        results[6] = KI.get_group_id(T) isa WorkItemNT{T}
    end
    return
end

# Records the typed indexing queries for every work-item, so the host can check
# that they agree with the default `Int` form across all three dimensions.
# `results` is `(work-items, 18)`: one row per work-item, holding the `x`, `y`
# and `z` components of each of the six queries in turn.
function typed_index_kernel(results, ::Type{T}) where {T}
    i, j, k = KI.get_global_id(T)
    ni, nj, _ = KI.get_global_size(T)
    lin = (k - one(T)) * ni * nj + (j - one(T)) * ni + i

    if lin <= size(results, 1)
        vals = (
            KI.get_global_size(T)..., KI.get_global_id(T)..., KI.get_local_size(T)...,
            KI.get_local_id(T)..., KI.get_num_groups(T)..., KI.get_group_id(T)...,
        )
        for q in 1:18
            @inbounds results[lin, q] = vals[q]
        end
    end
    return
end

# Records the `x` components of the typed queries in a type too small for the launch,
# indexed by the (untyped) global id.
function wrapping_index_kernel(results, ::Type{T}) where {T}
    i = KI.get_global_id().x
    if i <= size(results, 1)
        @inbounds begin
            results[i, 1] = KI.get_global_id(T).x
            results[i, 2] = KI.get_global_size(T).x
            results[i, 3] = KI.get_local_id(T).x
            results[i, 4] = KI.get_local_size(T).x
            results[i, 5] = KI.get_group_id(T).x
            results[i, 6] = KI.get_num_groups(T).x
        end
    end
    return
end

function subgroup_typecheck_kernel(results, val::T) where {T}
    # uniformly executed by the whole sub-group, as `shfl_down` requires
    shuffled = KI.shfl_down(val, 0x00000001)
    if KI.get_sub_group_local_id() == 1
        @inbounds begin
            results[1] = KI.get_sub_group_size() isa UInt32
            results[2] = KI.get_max_sub_group_size() isa UInt32
            results[3] = KI.get_num_sub_groups() isa UInt32
            results[4] = KI.get_sub_group_id() isa UInt32
            results[5] = KI.get_sub_group_local_id() isa UInt32
            results[6] = shuffled isa T
        end
    end
    return
end

function shfl_down_test_kernel(a, b, ::Val{N}) where {N}
    idx = KI.get_sub_group_local_id()

    val = a[idx]

    offset = 0x00000001
    while offset < N
        val += KI.shfl_down(val, offset)
        offset <<= 1
    end

    KI.sub_group_barrier()

    if idx == 1
        b[idx] = val
    end
    return
end

function interface_testsuite(backend::KI.Backend, AT)
    @testset "Launch parameters" begin
        # unequal group counts and sizes in every dimension, so that confusing them shows
        function run(dims; kwargs...)
            arr = KI.zeros(backend, Int32, dims)
            kernel = KI.@launch backend launch = false launch_kernel(arr)
            kernel(arr; kwargs...)
            KI.synchronize(backend)
            return Array(arr)
        end
        @test all(==(1), run((6, 1, 1); numgroups = 3, workgroupsize = 2))
        @test all(==(1), run((6, 1, 1); numgroups = (2,), workgroupsize = (3,)))
        @test all(==(1), run((6, 10, 1); numgroups = (2, 5), workgroupsize = (3, 2)))
        @test all(==(1), run((6, 10, 12); numgroups = (2, 5, 3), workgroupsize = (3, 2, 4)))

        # `ndrange` rounds up to whole groups, and doesn't mask the padding
        @test all(==(1), run((7, 5, 3); ndrange = (7, 5, 3), workgroupsize = (2, 3, 2)))
        @test all(==(1), run((7, 5, 3); ndrange = (7, 5, 3)))
        @test all(==(1), run((1000, 1, 1); ndrange = 1000))

        # defaults: one work-group of one work-item
        @test run((2, 2, 1)) == reshape(Int32[1, 0, 0, 0], 2, 2, 1)
        @test run((4, 1, 1); numgroups = 2) == reshape(Int32[1, 1, 0, 0], 4, 1, 1)
        @test run((4, 1, 1); workgroupsize = 2) == reshape(Int32[1, 1, 0, 0], 4, 1, 1)

        # nothing to launch
        @test all(==(0), run((2, 2, 2); ndrange = 0))
        @test all(==(0), run((2, 2, 2); ndrange = (2, 0)))
        @test all(==(0), run((2, 2, 2); numgroups = (2, 0, 2), workgroupsize = 2))

        # the global size is the padded ndrange
        results = AT(Vector{KernelData}(undef, 12))
        kernel = KI.@launch backend launch = false test_interface_kernel(results)
        kernel(results; ndrange = 10, workgroupsize = 4)
        KI.synchronize(backend)
        @test all(d -> d.global_size == 12 && d.num_groups == 3, Array(results))
    end

    @testset "Launch validation" begin
        arr = KI.zeros(backend, Int32, (1, 1, 1))
        kernel = KI.@launch backend launch = false launch_kernel(arr)
        max_items = KI.max_work_group_size(kernel)
        max_dims = KI.max_work_group_dims(backend)

        @test_throws ArgumentError kernel(arr; numgroups = (2, 2, 2, 2), workgroupsize = (2, 2, 2))
        @test_throws ArgumentError kernel(arr; numgroups = (2, 2, 2), workgroupsize = (2, 2, 2, 2))
        @test_throws ArgumentError kernel(arr; ndrange = (2, 2, 2, 2))
        @test_throws ArgumentError kernel(arr; ndrange = 4, numgroups = 2)
        @test_throws ArgumentError kernel(arr; workgroupsize = 0)
        @test_throws ArgumentError kernel(arr; workgroupsize = (2, 0))
        @test_throws ArgumentError kernel(arr; workgroupsize = -1)
        @test_throws ArgumentError kernel(arr; numgroups = -1)
        @test_throws ArgumentError kernel(arr; ndrange = (4, -1))
        @test_throws ArgumentError kernel(arr; ndrange = 4.0)
        @test_throws ArgumentError kernel(arr; ndrange = 4, max_work_group_size = 0)
        @test_throws ArgumentError kernel(arr; workgroupsize = max_items + 1)
        if max_dims[3] < max_items
            @test_throws ArgumentError kernel(arr; workgroupsize = (1, 1, max_dims[3] + 1))
        end
        KI.synchronize(backend)
        @test Array(arr) == zeros(Int32, 1, 1, 1)

        # other keywords are passed to the backend, which rejects the ones it doesn't know
        @test_throws Exception kernel(arr; this_is_not_a_launch_option = 1)
    end

    @testset "Launch limits" begin
        max_dims = KI.max_work_group_dims(backend)
        max_groups = KI.max_num_groups(backend)
        @test max_dims isa NTuple{3, Int} && all(>=(1), max_dims)
        @test max_groups isa NTuple{3, Int} && all(>=(1), max_groups)
        @test KI.max_work_group_size(backend) isa Int

        function fill_kernel(arr)
            i, j, k = KI.get_global_id()
            if i <= size(arr, 1) && j <= size(arr, 2) && k <= size(arr, 3)
                @inbounds arr[i, j, k] = 1.0f0
            end
            return
        end
        kernel = KI.@launch backend launch = false fill_kernel(AT(zeros(Float32, 1, 1, 1)))
        function fill_test(dims; kwargs...)
            arr = AT(zeros(Float32, dims))
            kernel(arr; kwargs...)
            KI.synchronize(backend)
            return all(Array(arr) .== 1)
        end

        max_items = KI.max_work_group_size(kernel)
        @test max_items isa Int && 1 <= max_items <= KI.max_work_group_size(backend)
        config = KI.launch_configuration(kernel)
        @test config isa @NamedTuple{workgroupsize::Int}
        @test 1 <= config.workgroupsize <= max_items
        @test KI.launch_configuration(kernel; max_work_group_size = 1).workgroupsize == 1
        # the recommendation is legal, whatever the size of the launch
        @testset "nitems = $nitems, max_work_group_size = $cap" for nitems in (nothing, 1, 1000, typemax(Int)),
                cap in (1, 7, typemax(Int))
            config = KI.launch_configuration(kernel; nitems, max_work_group_size = cap)
            @test 1 <= config.workgroupsize <= min(cap, max_items)
        end

        # automatically chosen workgroup sizes respect the per-dimension limits
        @testset "ndrange = $ndrange" for ndrange in ((1, 1, 5000), (1, 5000, 1), (1, 3, 2000))
            @test fill_test(ndrange; ndrange)
            @test fill_test(ndrange; ndrange, max_work_group_size = 7)
        end

        # the reported limits can be launched
        @testset "dimension $d" for d in 1:3
            items = min(max_dims[d], max_items)
            workgroupsize = ntuple(i -> i == d ? items : 1, 3)
            @test fill_test(workgroupsize; workgroupsize, numgroups = (1, 1, 1))

            # don't launch (practically) unlimited grids
            groups = min(max_groups[d], 2^16)
            numgroups = ntuple(i -> i == d ? groups : 1, 3)
            @test fill_test(numgroups; workgroupsize = (1, 1, 1), numgroups)
        end
    end

    @testset "Host return types" begin
        b = backend

        @test KI.supports_unified(b) isa Bool
        @test KI.supports_atomics(b) isa Bool
        @test KI.supports_float64(b) isa Bool
        @test KI.functional(b) isa Union{Missing, Bool}
        @test KI.multiprocessor_count(b) isa Int

        @test KI.device(b) isa Int
        @test KI.ndevices(b) isa Int
        # @test KI.device!(b, KI.device(b)) isa Nothing
        # @test KI.priority!(b, :normal) isa Nothing

        @test KI.shfl_down_types(b) isa Vector{DataType}

        arr = KI.allocate(b, Float32, 2)
        @test arr isa AT{Float32, 1}
        @test KI.zeros(b, Float32, 2) isa AT{Float32, 1}
        @test KI.ones(b, Float32, 2) isa AT{Float32, 1}
        @test KI.get_backend(arr) isa KI.Backend

    end

    @testset "Device return types" begin
        results = KI.zeros(backend, Bool, 6)
        KI.@launch backend typecheck_kernel(results)
        KI.synchronize(backend)
        @test all(Array(results))

        @testset "$T" for T in (Int32, Int64, UInt32, UInt64)
            typed_results = KI.zeros(backend, Bool, 6)
            KI.@launch backend typed_typecheck_kernel(typed_results, T)
            KI.synchronize(backend)
            @test all(Array(typed_results))
        end
    end

    @testset "Typed indexing" begin
        workgroupsize = (2, 3, 2)
        numgroups = (3, 2, 4)
        N = prod(workgroupsize) * prod(numgroups)

        # `Int` is the reference: it is what the zero-argument form returns.
        function run_typed(::Type{T}) where {T}
            results = KI.zeros(backend, T, N, 18)
            KI.@launch backend workgroupsize = workgroupsize numgroups = numgroups typed_index_kernel(results, T)
            KI.synchronize(backend)
            return Array(results)
        end
        reference = run_typed(Int)

        global_size = workgroupsize .* numgroups
        @test all(eachrow(reference[:, 1:3]) .== Ref(collect(global_size)))
        @test all(eachrow(reference[:, 7:9]) .== Ref(collect(workgroupsize)))
        @test all(eachrow(reference[:, 13:15]) .== Ref(collect(numgroups)))
        # every global id is seen exactly once, and agrees with the group and local ids
        @test sort(Tuple.(eachrow(reference[:, 4:6]))) == sort(vec(Tuple.(CartesianIndices(global_size))))
        @test reference[:, 4:6] == (reference[:, 16:18] .- 1) .* reference[:, 7:9] .+ reference[:, 10:12]

        @testset "$T" for T in (Int32, UInt32, Int64, UInt64)
            typed = run_typed(T)
            @test typed isa AbstractMatrix{T}
            @test typed == reference
        end
    end

    @testset "Typed indexing wraps" begin
        # a type too small for the launch wraps around (like `x % T`) instead of throwing
        @testset "$T" for (T, workgroupsize, numgroups) in (
                (UInt8, 128, 3), (Int8, 128, 3), (Int16, 256, 160),
            )
            N = workgroupsize * numgroups
            results = KI.zeros(backend, T, N, 6)
            KI.@launch backend workgroupsize = workgroupsize numgroups = numgroups wrapping_index_kernel(results, T)
            KI.synchronize(backend)
            results = Array(results)
            ids = 1:N
            @test results[:, 1] == ids .% T
            @test all(==(N % T), results[:, 2])
            @test results[:, 3] == mod1.(ids, workgroupsize) .% T
            @test all(==(workgroupsize % T), results[:, 4])
            @test results[:, 5] == cld.(ids, workgroupsize) .% T
            @test all(==(numgroups % T), results[:, 6])
        end
    end

    @testset "Basic interface functionality" begin
        workgroupsize = 4
        numgroups = 3
        N = workgroupsize * numgroups
        results = AT(Vector{KernelData}(undef, N))
        kernel = KI.@launch backend launch = false test_interface_kernel(results)

        kernel(results; workgroupsize, numgroups)
        KI.synchronize(backend)

        host_results = Array(results)
        for (i, k_data) in enumerate(host_results)
            @test k_data.global_id == i
            @test k_data.global_size == N
            @test k_data.local_size == workgroupsize
            @test k_data.num_groups == numgroups
            @test k_data.group_id == div(i - 1, workgroupsize) + 1
            @test k_data.local_id == ((i - 1) % workgroupsize) + 1
        end
    end

    # Used as a proxy for sub-group support
    if !isempty(KI.shfl_down_types(backend))
        @testset "Sub-group return types" begin
            @test KI.sub_group_size(backend) isa Int

            T = first(setdiff(KI.shfl_down_types(backend), [Bool]))
            results = KI.zeros(backend, Bool, 6)
            KI.@launch backend workgroupsize = KI.sub_group_size(backend) subgroup_typecheck_kernel(results, one(T))
            KI.synchronize(backend)
            @test all(Array(results))
        end

        @testset "Sub-groups" begin
            @test KI.sub_group_size(backend) isa Int

            # Test with small kernel
            sg_size = KI.sub_group_size(backend)
            sg_n = 2
            workgroupsize = sg_size * sg_n
            numgroups = 2
            N = workgroupsize * numgroups

            results = AT(Vector{SubgroupData}(undef, N))
            kernel = KI.@launch backend launch = false test_subgroup_kernel(results)

            kernel(results; workgroupsize, numgroups)
            KI.synchronize(backend)

            host_results = Array(results)

            # Verify results make sense
            for (i, sg_data) in enumerate(host_results)
                @test sg_data.sub_group_size == sg_size
                @test sg_data.max_sub_group_size == sg_size
                @test sg_data.num_sub_groups == sg_n

                # Group ID should be 1-based
                expected_sub_group = div(((i - 1) % workgroupsize), sg_size) + 1
                @test sg_data.sub_group_id == expected_sub_group

                # Local ID should be 1-based within group
                expected_sg_local = ((i - 1) % sg_size) + 1
                @test sg_data.sub_group_local_id == expected_sg_local
            end
        end
        @testset "shfl_down" begin
            @test !isempty(KI.shfl_down_types(backend))
            types_to_test = setdiff(KI.shfl_down_types(backend), [Bool])
            @testset "$T" for T in types_to_test
                N = KI.sub_group_size(backend)
                a = zeros(T, N)
                rand!(a, (0:1))

                dev_a = AT(a)
                dev_b = AT(zeros(T, N))

                KI.@launch backend workgroupsize = N shfl_down_test_kernel(dev_a, dev_b, Val(N))

                b = Array(dev_b)
                @test sum(a) ≈ b[1]
            end
        end
    end
    return nothing
end
