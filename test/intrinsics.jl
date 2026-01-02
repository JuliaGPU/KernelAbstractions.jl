import KernelAbstractions.KernelIntrinsics as KI

struct KernelData
    global_size::Int
    global_id::Int
    local_size::Int
    local_id::Int
    num_groups::Int
    group_id::Int
end
function test_intrinsics_kernel(results)
    i = KI.get_global_id().x

    if i <= length(results)
        @inbounds results[i] = KernelData(KI.get_global_size().x,
                                          KI.get_global_id().x,
                                          KI.get_local_size().x,
                                          KI.get_local_id().x,
                                          KI.get_num_groups().x,
                                          KI.get_group_id().x)
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
        @inbounds results[i] = SubgroupData(KI.get_sub_group_size(),
                                            KI.get_max_sub_group_size(),
                                            KI.get_num_sub_groups(),
                                            KI.get_sub_group_id(),
                                            KI.get_sub_group_local_id())
    end
    return
end

# Do NOT use this kernel as an example for your code.
#  It was written assuming one workgroup of size 32 and
#  is only valid for those
function shfl_down_test_kernel(a, b, ::Val{N}) where N
    # This is not valid
    idx = KI.get_sub_group_local_id()

    temp = KI.localmemory(eltype(b), N)
    temp[idx] = a[idx]

    KI.barrier()

    if idx == 1
        value = temp[idx]

        if KI.get_sub_group_size() > 32
            value = value + KI.shfl_down(value, 32)
            KI.sub_group_barrier()
        end
        value = value + KI.shfl_down(value, 16)
        KI.sub_group_barrier()

        value = value + KI.shfl_down(value,  8)
        KI.sub_group_barrier()

        value = value + KI.shfl_down(value,  4)
        KI.sub_group_barrier()

        value = value + KI.shfl_down(value,  2)
        KI.sub_group_barrier()

        value = value + KI.shfl_down(value,  1)
        KI.sub_group_barrier()

        b[idx] = value
    end
    return
end

function intrinsics_testsuite(backend, AT)
    @testset "KernelIntrinsics Tests" begin
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
            KernelAbstractions.synchronize(backend())
            @test all(Array(arr1d) .== 1)

            # 1d tuple
            arr1dt = AT(zeros(Float32, 4))
            KI.@kernel backend() numworkgroups = (2,) workgroupsize = (2,) launch_kernel1d(arr1dt)
            KernelAbstractions.synchronize(backend())
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
            KernelAbstractions.synchronize(backend())
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
            KernelAbstractions.synchronize(backend())
            @test all(Array(arr3d) .== 1)

            # 4d (Errors)
            @test_throws ArgumentError (KI.@kernel backend() numworkgroups = (2, 2, 2, 2) workgroupsize = (2, 2, 2) launch_kernel3d(arr3d))
            @test_throws ArgumentError (KI.@kernel backend() numworkgroups = (2, 2, 2) workgroupsize = (2, 2, 2, 2) launch_kernel3d(arr3d))
        end

        @testset "Basic intrinsics functionality" begin

            @test KI.max_work_group_size(backend()) isa Int
            @test KI.multiprocessor_count(backend()) isa Int

            # Test with small kernel
            workgroupsize = 4
            numworkgroups = 4
            N = workgroupsize * numworkgroups
            results = AT(Vector{KernelData}(undef, N))
            kernel = KI.@kernel backend() launch = false test_intrinsics_kernel(results)

            @test KI.kernel_max_work_group_size(kernel) isa Int
            @test KI.kernel_max_work_group_size(kernel; max_work_items = 1) == 1

            kernel(results; workgroupsize, numworkgroups)
            KernelAbstractions.synchronize(backend())

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

        @testset "Subgroups" begin
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
            KernelAbstractions.synchronize(backend())

            host_results = Array(results)

            # Verify results make sense
            for (i, sg_data) in enumerate(host_results)
                @test sg_data.sub_group_size == sg_size
                @test sg_data.max_sub_group_size == sg_size
                @test sg_data.num_sub_groups == sg_n

                # Group ID should be 1-based
                div(((i - 1) % workgroupsize), sg_n) + 1
                expected_sub_group = div(((i - 1) % workgroupsize), sg_size) + 1
                @test sg_data.sub_group_id == expected_sub_group

                # Local ID should be 1-based within group
                expected_sg_local = ((i - 1) % sg_size) + 1
                @test sg_data.sub_group_local_id == expected_sg_local
            end
        end
        @testset "shfl_down(::$T)" for T in KI.shfl_down_types(backend())
            N = KI.sub_group_size(backend())
            a = zeros(T, N)
            rand!(a, (1:4))

            dev_a = AT(a)
            dev_b = AT(zeros(T, N))

            KI.@kernel backend() workgroupsize=N shfl_down_test_kernel(dev_a, dev_b, Val(N))

            b = Array(dev_b)
            @test sum(a) ≈ b[1]
        end
    end
    return nothing
end
