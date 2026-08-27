import KernelInterface as KI
using Random

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
const WorkItemNT = @NamedTuple{x::Int, y::Int, z::Int}

function typecheck_kernel(results)
    @inbounds begin
        results[1] = KI.get_global_size() isa WorkItemNT
        results[2] = KI.get_global_id() isa WorkItemNT
        results[3] = KI.get_local_size() isa WorkItemNT
        results[4] = KI.get_local_id() isa WorkItemNT
        results[5] = KI.get_num_groups() isa WorkItemNT
        results[6] = KI.get_group_id() isa WorkItemNT
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

function interface_testsuite(backend, AT)
    @testset "Launch parameters" begin
        # 1d
        function launch_kernel1d(arr)
            i, _, _ = KI.get_local_id()
            gi, _, _ = KI.get_group_id()
            ngi, _, _ = KI.get_num_groups()

            arr[(gi - 1) * ngi + i] = 1.0f0
            return
        end
        arr1d = AT(zeros(Float32, 4))
        KI.@kernel backend() numworkgroups = 2 workgroupsize = 2 launch_kernel1d(arr1d)
        KI.synchronize(backend())
        @test all(Array(arr1d) .== 1)

        # 1d tuple
        arr1dt = AT(zeros(Float32, 4))
        KI.@kernel backend() numworkgroups = (2,) workgroupsize = (2,) launch_kernel1d(arr1dt)
        KI.synchronize(backend())
        @test all(Array(arr1dt) .== 1)

        # 2d
        function launch_kernel2d(arr)
            i, j, _ = KI.get_local_id()
            gi, gj, _ = KI.get_group_id()
            ngi, ngj, _ = KI.get_num_groups()

            arr[(gi - 1) * ngi + i, (gj - 1) * ngj + j] = 1.0f0
            return
        end
        arr2d = AT(zeros(Float32, 4, 4))
        KI.@kernel backend() numworkgroups = (2, 2) workgroupsize = (2, 2) launch_kernel2d(arr2d)
        KI.synchronize(backend())
        @test all(Array(arr2d) .== 1)

        # 3d
        function launch_kernel3d(arr)
            i, j, k = KI.get_local_id()
            gi, gj, gk = KI.get_group_id()
            ngi, ngj, ngk = KI.get_num_groups()

            arr[(gi - 1) * ngi + i, (gj - 1) * ngj + j, (gk - 1) * ngk + k] = 1.0f0
            return
        end
        arr3d = AT(zeros(Float32, 4, 4, 4))
        KI.@kernel backend() numworkgroups = (2, 2, 2) workgroupsize = (2, 2, 2) launch_kernel3d(arr3d)
        KI.synchronize(backend())
        @test all(Array(arr3d) .== 1)

        # 4d (Errors)
        @test_throws ArgumentError (KI.@kernel backend() numworkgroups = (2, 2, 2, 2) workgroupsize = (2, 2, 2) launch_kernel3d(arr3d))
        @test_throws ArgumentError (KI.@kernel backend() numworkgroups = (2, 2, 2) workgroupsize = (2, 2, 2, 2) launch_kernel3d(arr3d))
    end

    @testset "Host return types" begin
        b = backend()

        @test KI.supports_unified(b) isa Bool
        @test KI.supports_atomics(b) isa Bool
        @test KI.supports_float64(b) isa Bool
        @test KI.functional(b) isa Union{Missing, Bool}

        @test KI.device(b) isa Int
        @test KI.ndevices(b) isa Int
        @test KI.device!(b, KI.device(b)) isa Nothing
        @test KI.priority!(b, :normal) isa Nothing

        @test KI.shfl_down_types(b) isa Vector{DataType}

        arr = KI.allocate(b, Float32, 2)
        @test arr isa AT{Float32, 1}
        @test KI.zeros(b, Float32, 2) isa AT{Float32, 1}
        @test KI.ones(b, Float32, 2) isa AT{Float32, 1}
        @test KI.get_backend(arr) isa KI.Backend

    end

    @testset "Device return types" begin
        results = KI.zeros(backend(), Bool, 6)
        KI.@kernel backend() typecheck_kernel(results)
        KI.synchronize(backend())
        @test all(Array(results))
    end

    @testset "Basic interface functionality" begin

        @test KI.max_work_group_size(backend()) isa Int
        @test KI.multiprocessor_count(backend()) isa Int

        # Test with small kernel
        workgroupsize = 4
        numworkgroups = 4
        N = workgroupsize * numworkgroups
        results = AT(Vector{KernelData}(undef, N))
        kernel = KI.@kernel backend() launch = false test_interface_kernel(results)

        @test KI.kernel_max_work_group_size(kernel) isa Int
        @test KI.kernel_max_work_group_size(kernel; max_work_items = 1) == 1

        kernel(results; workgroupsize, numworkgroups)
        KI.synchronize(backend())

        host_results = Array(results)

        # Verify results make sense
        for (i, k_data) in enumerate(host_results)

            # Global IDs should be 1-based and sequential
            @test k_data.global_id == i

            # Global size should match our ndrange
            @test k_data.global_size == N

            @test k_data.local_size == workgroupsize

            @test k_data.num_groups == numworkgroups

            # Group ID should be 1-based
            expected_group = div(i - 1, numworkgroups) + 1
            @test k_data.group_id == expected_group

            # Local ID should be 1-based within group
            expected_local = ((i - 1) % workgroupsize) + 1
            @test k_data.local_id == expected_local
        end
    end

    # Used as a proxy for sub-group support
    if !isempty(KI.shfl_down_types(backend()))
        @testset "Sub-group return types" begin
            @test KI.sub_group_size(backend()) isa Int

            T = first(setdiff(KI.shfl_down_types(backend()), [Bool]))
            results = KI.zeros(backend(), Bool, 6)
            KI.@kernel backend() workgroupsize = KI.sub_group_size(backend()) subgroup_typecheck_kernel(results, one(T))
            KI.synchronize(backend())
            @test all(Array(results))
        end

        @testset "Sub-groups" begin
            @test KI.sub_group_size(backend()) isa Int

            # Test with small kernel
            sg_size = KI.sub_group_size(backend())
            sg_n = 2
            workgroupsize = sg_size * sg_n
            numworkgroups = 2
            N = workgroupsize * numworkgroups

            results = AT(Vector{SubgroupData}(undef, N))
            kernel = KI.@kernel backend() launch = false test_subgroup_kernel(results)

            kernel(results; workgroupsize, numworkgroups)
            KI.synchronize(backend())

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
            @test !isempty(KI.shfl_down_types(backend()))
            types_to_test = setdiff(KI.shfl_down_types(backend()), [Bool])
            @testset "$T" for T in types_to_test
                N = KI.sub_group_size(backend())
                a = zeros(T, N)
                rand!(a, (0:1))

                dev_a = AT(a)
                dev_b = AT(zeros(T, N))

                KI.@kernel backend() workgroupsize = N shfl_down_test_kernel(dev_a, dev_b, Val(N))

                b = Array(dev_b)
                @test sum(a) ≈ b[1]
            end
        end
    end
    return nothing
end
