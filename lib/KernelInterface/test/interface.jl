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
# Local arrays, written by one work-item and read back by another one after a barrier.
# `c` has the same type and size as `a`, but is a different call site, so different memory.
function localmem_kernel(out, ::Val{N}) where {N}
    i = KI.get_local_id().x
    a = KI.localmemory(Int32, N)
    b = KI.localmemory(Int32, (2, N))
    c = KI.localmemory(Int32, N)
    @inbounds begin
        a[i] = i
        b[1, i] = -i
        b[2, i] = 2i
        c[i] = 3i
    end
    KI.barrier()
    j = N - i + 1
    gid = KI.get_global_id().x
    @inbounds begin
        out[gid, 1] = a[j]
        out[gid, 2] = b[1, j]
        out[gid, 3] = b[2, j]
        out[gid, 4] = c[j]
    end
    return
end

# Every work-item writes global memory, and reads another work-item's write after a barrier.
function global_barrier_kernel(scratch, out)
    n = KI.get_local_size().x
    base = (KI.get_group_id().x - 1) * n
    i = KI.get_local_id().x
    @inbounds scratch[base + i] = base + i
    KI.barrier()
    @inbounds out[base + i] = scratch[base + n - i + 1]
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

struct SubgroupData
    sub_group_size::Int
    max_sub_group_size::Int
    num_sub_groups::Int
    sub_group_id::Int
    sub_group_local_id::Int
end
function test_subgroup_kernel(results)
    l = KI.get_local_id()
    s = KI.get_local_size()
    i = (l.y - 1) * s.x + l.x + (KI.get_group_id().x - 1) * s.x * s.y

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

function subgroup_typecheck_kernel(results, val::T) where {T}
    # uniformly executed by the whole sub-group, as `shfl_down` requires
    shuffled = KI.shfl_down(val, 1)
    if KI.get_sub_group_local_id() == 1
        @inbounds begin
            results[1] = KI.get_sub_group_size() isa Int
            results[2] = KI.get_max_sub_group_size() isa Int
            results[3] = KI.get_num_sub_groups() isa Int
            results[4] = KI.get_sub_group_id() isa Int
            results[5] = KI.get_sub_group_local_id() isa Int
            results[6] = shuffled isa T
            results[7] = KI.get_sub_group_size(UInt32) isa UInt32
            results[8] = KI.get_max_sub_group_size(Int32) isa Int32
            results[9] = KI.get_num_sub_groups(UInt32) isa UInt32
            results[10] = KI.get_sub_group_id(Int32) isa Int32
            results[11] = KI.get_sub_group_local_id(UInt32) isa UInt32
        end
    end
    return
end

function shfl_down_test_kernel(a, b, ::Val{N}) where {N}
    idx = KI.get_sub_group_local_id()

    val = a[idx]

    # the result of shuffling from a lane past the end is unspecified, so don't add it
    offset = 1
    while offset < N
        shuffled = KI.shfl_down(val, offset)
        if idx + offset <= N
            val += shuffled
        end
        offset <<= 1
    end

    KI.sub_group_barrier()

    if idx == 1
        b[idx] = val
    end
    return
end

# Every lane shuffles its lane id down by `offset`; `out` gets the lane id, the sub-group
# size and the result.
function shfl_down_lanes_kernel(out, ::Type{T}, offset) where {T}
    lane = KI.get_sub_group_local_id()
    shuffled = KI.shfl_down(T(lane), offset)
    i = KI.get_global_id().x
    @inbounds begin
        out[i, 1] = lane
        out[i, 2] = KI.get_sub_group_size()
        out[i, 3] = shuffled
    end
    return
end

# Every lane writes local and global memory, and reads another lane's write after a
# sub-group barrier. `N` is the sub-group width; the work-group is one sub-group.
function sub_group_barrier_kernel(scratch, out, ::Val{N}) where {N}
    lane = KI.get_sub_group_local_id()
    other = mod1(lane + 1, KI.get_sub_group_size())
    lm = KI.localmemory(Int32, N)
    @inbounds lm[lane] = lane
    @inbounds scratch[lane] = -lane
    KI.sub_group_barrier()
    @inbounds out[lane, 1] = lm[other]
    @inbounds out[lane, 2] = scratch[other]
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
        @test KI.supports_subgroups(b) isa Bool
        @test KI.supports_shuffle(b, Float32) isa Bool
        @test KI.functional(b) isa Union{Missing, Bool}
        @test KI.multiprocessor_count(b) isa Int

        @test KI.device(b) isa Int
        @test KI.ndevices(b) isa Int

        arr = KI.allocate(b, Float32, 2)
        @test arr isa AT{Float32, 1}
        @test KI.zeros(b, Float32, 2) isa AT{Float32, 1}
        @test KI.ones(b, Float32, 2) isa AT{Float32, 1}
        @test KI.get_backend(arr) isa KI.Backend
        @test KI.device(b, arr) == KI.device(b)

        kernel = KI.@launch b launch = false test_interface_kernel(AT(Vector{KernelData}(undef, 1)))
        @test kernel isa KI.Kernel
        @test kernel.backend === b
    end

    @testset "Devices" begin
        b = backend
        dev = KI.device(b)
        @test 1 <= dev <= KI.ndevices(b)
        KI.device!(b, dev)
        @test KI.device(b) == dev
        @test_throws ArgumentError KI.device!(b, 0)
        @test_throws ArgumentError KI.device!(b, KI.ndevices(b) + 1)

        if KI.ndevices(b) > 1
            other = mod1(dev + 1, KI.ndevices(b))
            arr_dev = KI.ones(b, Float32, 4)
            try
                KI.device!(b, other)
                # the owner of an array doesn't change with the active device
                @test KI.device(b, arr_dev) == dev
                @test KI.device(b) == other
                arr = KI.ones(b, Float32, 4)
                @test KI.device(b, arr) == other
                @test Array(arr) == ones(Float32, 4)
            finally
                KI.device!(b, dev)
            end
            @test KI.device(b) == dev
        end
    end

    @testset "copyto!" begin
        b = backend
        host = rand(Float32, 16)
        dev = KI.allocate(b, Float32, 16)
        @test KI.copyto!(b, dev, host) === dev
        dev2 = KI.allocate(b, Float32, 16)
        @test KI.copyto!(b, dev2, dev) === dev2
        back = zeros(Float32, 16)
        @test KI.copyto!(b, back, dev2) === back
        KI.synchronize(b)
        @test back == host

        # arrays of different shapes copy by linear index
        dev3 = KI.allocate(b, Float32, (4, 4))
        KI.copyto!(b, dev3, host)
        KI.synchronize(b)
        @test Array(dev3) == reshape(host, 4, 4)

        # ordered with respect to kernels on the same queue
        arr = KI.zeros(b, Int32, (4, 1, 1))
        KI.@launch b numgroups = 1 workgroupsize = 4 launch_kernel(arr)
        result = zeros(Int32, 4)
        KI.copyto!(b, result, arr)
        KI.synchronize(b)
        @test result == ones(Int32, 4)

        @test_throws ArgumentError KI.copyto!(b, KI.allocate(b, Float32, 8), host)
        @test_throws ArgumentError KI.copyto!(b, zeros(Float32, 8), dev)
        @test_throws ArgumentError KI.copyto!(b, KI.allocate(b, Float32, 32), dev)
        KI.synchronize(b)
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

    @testset "Local memory and barriers" begin
        N = 32
        groups = 3
        out = KI.zeros(backend, Int32, N * groups, 4)
        KI.@launch backend numgroups = groups workgroupsize = N localmem_kernel(out, Val(N))
        KI.synchronize(backend)
        out = Array(out)
        # each work-item sees what its mirror image wrote before the barrier, in both arrays
        expected = repeat(N:-1:1, groups)
        @test out[:, 1] == expected
        @test out[:, 2] == -expected
        @test out[:, 3] == 2 .* expected
        @test out[:, 4] == 3 .* expected

        scratch = KI.zeros(backend, Int, N * groups)
        out = KI.zeros(backend, Int, N * groups)
        KI.@launch backend numgroups = groups workgroupsize = N global_barrier_kernel(scratch, out)
        KI.synchronize(backend)
        @test Array(out) == vcat([(g * N) .+ (N:-1:1) for g in 0:(groups - 1)]...)
    end

    if KI.supports_subgroups(backend)
        sg_size = KI.sub_group_size(backend)
        max_dims = KI.max_work_group_dims(backend)
        # whether `kernel` can be launched with work-groups of size `dims`
        fits(kernel, dims) = all(dims .<= max_dims[1:length(dims)]) && prod(dims) <= KI.max_work_group_size(kernel)

        @testset "Sub-group return types" begin
            @test sg_size isa Int && sg_size >= 1

            types = filter(T -> KI.supports_shuffle(backend, T), [Int32, Float32, Int64, UInt32])
            if !isempty(types)
                results = KI.zeros(backend, Bool, 11)
                val = one(first(types))
                kernel = KI.@launch backend launch = false subgroup_typecheck_kernel(results, val)
                if fits(kernel, (sg_size,))
                    kernel(results, val; workgroupsize = sg_size)
                    KI.synchronize(backend)
                    @test all(Array(results))
                else
                    @test_skip "work-groups of $sg_size work-items"
                end
            end
        end

        # checks the sub-groups of a work-group of `items` work-items
        function check_subgroups(data, items)
            @test all(d -> d.max_sub_group_size == sg_size, data)
            @test all(d -> d.num_sub_groups == cld(items, sg_size), data)
            @test all(d -> 1 <= d.sub_group_id <= cld(items, sg_size), data)
            @test all(d -> 1 <= d.sub_group_local_id <= d.sub_group_size, data)
            # every work-item has its own (sub-group, lane) pair
            @test allunique(map(d -> (d.sub_group_id, d.sub_group_local_id), data))
            # each sub-group has as many members as its size says
            for id in unique(map(d -> d.sub_group_id, data))
                members = filter(d -> d.sub_group_id == id, data)
                @test all(d -> d.sub_group_size == length(members), members)
            end
            return
        end

        @testset "Sub-groups" begin
            sg_n = 2
            workgroupsize = sg_size * sg_n
            numgroups = 2
            N = workgroupsize * numgroups

            results = AT(Vector{SubgroupData}(undef, N))
            kernel = KI.@launch backend launch = false test_subgroup_kernel(results)
            if fits(kernel, (workgroupsize,))
                kernel(results; workgroupsize, numgroups)
                KI.synchronize(backend)

                host_results = Array(results)
                @test all(d -> d.sub_group_size == sg_size, host_results)
                for group in Iterators.partition(host_results, workgroupsize)
                    check_subgroups(collect(group), workgroupsize)
                end
            else
                @test_skip "work-groups of $workgroupsize work-items"
            end
        end

        @testset "Partial sub-groups" begin
            # a 2-D work-group whose size isn't a multiple of the sub-group size, or else a
            # 1-D one with one work-item more than a sub-group
            numgroups = 2
            results = AT(Vector{SubgroupData}(undef, max(66, sg_size + 1) * numgroups))
            kernel = KI.@launch backend launch = false test_subgroup_kernel(results)
            workgroupsize = fits(kernel, (33, 2)) ? (33, 2) : (sg_size + 1,)
            items = prod(workgroupsize)
            if fits(kernel, workgroupsize) && items % sg_size != 0
                kernel(results; workgroupsize, numgroups)
                KI.synchronize(backend)

                host_results = Array(results)[1:(items * numgroups)]
                for group in Iterators.partition(host_results, items)
                    group = collect(group)
                    check_subgroups(group, items)
                    # the sizes of the sub-groups add up to the work-group, with one partial one
                    sizes = Dict(d.sub_group_id => d.sub_group_size for d in group)
                    @test sum(values(sizes)) == items
                    @test count(<(sg_size), values(sizes)) == (items % sg_size == 0 ? 0 : 1)
                end
            else
                @test_skip "work-groups of $workgroupsize work-items"
            end
        end

        @testset "sub_group_barrier" begin
            out = KI.zeros(backend, Int32, sg_size, 2)
            scratch = KI.zeros(backend, Int32, sg_size)
            kernel = KI.@launch backend launch = false sub_group_barrier_kernel(scratch, out, Val(sg_size))
            if fits(kernel, (sg_size,))
                kernel(scratch, out, Val(sg_size); workgroupsize = sg_size)
                KI.synchronize(backend)
                other = mod1.(2:(sg_size + 1), sg_size)
                @test Array(out) == hcat(other, -other)
            else
                @test_skip "work-groups of $sg_size work-items"
            end
        end

        @testset "shfl_down" begin
            candidates = (
                Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
                Float16, Float32, Float64,
            )
            types = filter(T -> KI.supports_shuffle(backend, T), candidates)
            @testset "$T" for T in types
                a = zeros(T, sg_size)
                rand!(a, (0:1))
                dev_a = AT(a)
                dev_b = AT(zeros(T, sg_size))
                KI.@launch backend workgroupsize = sg_size shfl_down_test_kernel(dev_a, dev_b, Val(sg_size))
                KI.synchronize(backend)
                @test sum(a) ≈ Array(dev_b)[1]

                # every lane whose source is in range gets the source's value
                @testset "offset $offset" for offset in unique((1, 3, sg_size ÷ 2))
                    1 <= offset < sg_size || continue
                    N = 2 * sg_size
                    out = KI.zeros(backend, T, N, 3)
                    kernel = KI.@launch backend launch = false shfl_down_lanes_kernel(out, T, offset)
                    # one sub-group if two don't fit in a work-group
                    fits(kernel, (N,)) || (N = sg_size)
                    kernel(out, T, offset; workgroupsize = N)
                    KI.synchronize(backend)
                    out = Array(out)
                    in_range = findall(i -> out[i, 1] + offset <= out[i, 2], 1:N)
                    @test length(in_range) == N - (N ÷ sg_size) * offset
                    @test out[in_range, 3] == out[in_range, 1] .+ offset
                end
            end
        end
    end
    return nothing
end
