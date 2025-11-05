
function test_intrinsics_kernel(results)
    # Test all intrinsics return NamedTuples with x, y, z fields
    global_size = KI.get_global_size()
    global_id = KI.get_global_id()
    local_size = KI.get_local_size()
    local_id = KI.get_local_id()
    num_groups = KI.get_num_groups()
    group_id = KI.get_group_id()

    if UInt32(global_id.x) <= UInt32(global_size.x)
        results[1, global_id.x] = global_id.x
        results[2, global_id.x] = local_id.x
        results[3, global_id.x] = group_id.x
        results[4, global_id.x] = global_size.x
        results[5, global_id.x] = local_size.x
        results[6, global_id.x] = num_groups.x
    end
    return
end


function intrinsics_testsuite(backend, AT)
    @testset "KernelIntrinsics Tests" begin
        @testset "Basic intrinsics functionality" begin

            @test KI.max_work_group_size(backend()) isa Int
            @test KI.multiprocessor_count(backend()) isa Int

            # Test with small kernel
            N = 16
            results = AT(zeros(Int, 6, N))
            kernel = KI.@kikernel backend() launch=false test_intrinsics_kernel(results)

            @test KI.kernel_max_work_group_size(backend(), kernel) isa Int
            @test KI.kernel_max_work_group_size(backend(), kernel; max_work_items=1) == 1

            kernel(results, workgroupsize = 4, numworkgroups = 4)
            KernelAbstractions.synchronize(backend())

            host_results = Array(results)

            # Verify results make sense
            for i in 1:N
                global_id_x, local_id_x, group_id_x, global_size_x, local_size_x, num_groups_x = host_results[:, i]

                # Global IDs should be 1-based and sequential
                @test global_id_x == i

                # Global size should match our ndrange
                @test global_size_x == N

                # Local size should be 4 (our workgroupsize)
                @test local_size_x == 4

                # Number of groups should be ceil(N/4) = 4
                @test num_groups_x == 4

                # Group ID should be 1-based
                expected_group = div(i - 1, 4) + 1
                @test group_id_x == expected_group

                # Local ID should be 1-based within group
                expected_local = ((i - 1) % 4) + 1
                @test local_id_x == expected_local
            end
        end
    end
end
