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

# ── vload / vstore! kernel helpers ──────────────────────────────────────────

function vload_copy_kernel(dst, src, ::Val{N}) where {N}
    g   = KI.get_global_id().x
    idx = (g - 1) * N + 1
    KI.vstore!(dst, idx, KI.vload(Val(N), src, idx))
    return
end

function vload_scale_kernel(dst, src, scale::T, ::Val{N}) where {T, N}
    g   = KI.get_global_id().x
    idx = (g - 1) * N + 1
    vals = KI.vload(Val(N), src, idx)
    KI.vstore!(dst, idx, ntuple(i -> vals[i] * scale, Val(N)))
    return
end

function vload_reduce_kernel(dst, src, ::Val{N}) where {N}
    g    = KI.get_global_id().x
    idx  = (g - 1) * N + 1
    vals = KI.vload(Val(N), src, idx)
    s    = vals[1]
    for i in 2:N
        s += vals[i]
    end
    @inbounds dst[g] = s
    return
end

function vload_tail_kernel(dst, src, n, ::Val{N}) where {N}
    g   = KI.get_global_id().x
    idx = (g - 1) * N + 1
    if idx + N - 1 <= n
        KI.vstore!(dst, idx, KI.vload(Val(N), src, idx))
    else
        while idx <= n
            @inbounds dst[idx] = src[idx]
            idx += 1
        end
    end
    return
end

function vload_store_load_kernel(arr, v0::T, v1::T, v2::T, v3::T) where {T}
    KI.vstore!(arr, 1, (v0, v1, v2, v3))
    loaded = KI.vload(Val(4), arr, 1)
    KI.vstore!(arr, 5, loaded)
    return
end

function vload_mt_copy_kernel(dst, src, ::Val{N}) where {N}
    local_id  = KI.get_local_id().x
    group_id  = KI.get_group_id().x
    groupsize = KI.get_local_size().x
    idx = ((group_id - 1) * groupsize + (local_id - 1)) * N + 1
    KI.vstore!(dst, idx, KI.vload(Val(N), src, idx))
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

        @testset "vload / vstore!" begin

            fp64 = KernelAbstractions.supports_float64(backend())

            # ── 1. Roundtrip for every supported (N, T) combination ─────────
            vload_types = fp64 ? (Float32, Float64, Int32, Int64) : (Float32, Int32, Int64)
            @testset "roundtrip N=$N T=$T" for N in (1, 2, 4, 8),
                                               T in vload_types
                n   = 64
                src = AT(T.(1:n))
                dst = AT(zeros(T, n))
                KI.@kernel backend() numworkgroups = n ÷ N workgroupsize = 1 vload_copy_kernel(dst, src, Val(N))
                KernelAbstractions.synchronize(backend())
                @test Array(dst) == Array(src)
            end

            # ── 2. N=1 degenerate (scalar vload/vstore!) ────────────────────
            @testset "N=1 scalar degenerate" begin
                src = AT(Float32[42.0f0])
                dst = AT(Float32[0.0f0])
                KI.@kernel backend() numworkgroups = 1 workgroupsize = 1 vload_copy_kernel(dst, src, Val(1))
                KernelAbstractions.synchronize(backend())
                @test Array(dst) == Array(src)
            end

            # ── 3. Minimal array: size == N (one vload covers whole array) ──
            @testset "minimal n=N=$N" for N in (1, 2, 4)
                src = AT(Float32.(1:N))
                dst = AT(zeros(Float32, N))
                KI.@kernel backend() numworkgroups = 1 workgroupsize = 1 vload_copy_kernel(dst, src, Val(N))
                KernelAbstractions.synchronize(backend())
                @test Array(dst) == Array(src)
            end

            # ── 4. Arithmetic on loaded values ──────────────────────────────
            @testset "scale by 2 N=4 Float32" begin
                n   = 64
                src = AT(Float32.(1:n))
                dst = AT(zeros(Float32, n))
                KI.@kernel backend() numworkgroups = n ÷ 4 workgroupsize = 1 vload_scale_kernel(dst, src, 2.0f0, Val(4))
                KernelAbstractions.synchronize(backend())
                @test Array(dst) ≈ 2.0f0 .* Float32.(1:n)
            end

            if fp64
                @testset "scale by 3 N=2 Float64" begin
                    n   = 64
                    src = AT(Float64.(1:n))
                    dst = AT(zeros(Float64, n))
                    KI.@kernel backend() numworkgroups = n ÷ 2 workgroupsize = 1 vload_scale_kernel(dst, src, 3.0, Val(2))
                    KernelAbstractions.synchronize(backend())
                    @test Array(dst) ≈ 3.0 .* Float64.(1:n)
                end
            end

            # ── 5. Per-thread reduction: each thread sums its N elements ────
            @testset "per-thread sum all-ones N=4" begin
                n     = 64
                src   = AT(ones(Float32, n))
                result = AT(zeros(Float32, n ÷ 4))
                KI.@kernel backend() numworkgroups = n ÷ 4 workgroupsize = 1 vload_reduce_kernel(result, src, Val(4))
                KernelAbstractions.synchronize(backend())
                @test all(Array(result) .== 4.0f0)
            end

            @testset "per-thread sum sequential N=4" begin
                n      = 64
                src    = AT(Float32.(1:n))
                result = AT(zeros(Float32, n ÷ 4))
                KI.@kernel backend() numworkgroups = n ÷ 4 workgroupsize = 1 vload_reduce_kernel(result, src, Val(4))
                KernelAbstractions.synchronize(backend())
                expected = [Float32(sum(4k+1:4k+4)) for k in 0:n÷4-1]
                @test Array(result) == expected
            end

            # ── 6. Scalar tail: n not divisible by N ────────────────────────
            @testset "scalar tail n=$n" for n in (65, 66, 67)
                src = AT(Float32.(1:n))
                dst = AT(zeros(Float32, n))
                KI.@kernel backend() numworkgroups = cld(n, 4) workgroupsize = 1 vload_tail_kernel(dst, src, n, Val(4))
                KernelAbstractions.synchronize(backend())
                @test Array(dst) == Array(src)
            end

            # ── 7. vstore! → vload identity (store then read back) ──────────
            @testset "vstore then vload identity" begin
                arr = AT(zeros(Float32, 8))
                KI.@kernel backend() numworkgroups = 1 workgroupsize = 1 vload_store_load_kernel(arr, 10.0f0, 20.0f0, 30.0f0, 40.0f0)
                KernelAbstractions.synchronize(backend())
                h = Array(arr)
                @test h[1:4] == [10.0f0, 20.0f0, 30.0f0, 40.0f0]
                @test h[5:8] == h[1:4]
            end

            # ── 8. All-zeros and all-ones arrays ────────────────────────────
            @testset "all-zeros roundtrip N=4" begin
                n   = 64
                src = AT(zeros(Float32, n))
                dst = AT(ones(Float32, n))
                KI.@kernel backend() numworkgroups = n ÷ 4 workgroupsize = 1 vload_copy_kernel(dst, src, Val(4))
                KernelAbstractions.synchronize(backend())
                @test all(Array(dst) .== 0.0f0)
            end

            # ── 9. Large array ───────────────────────────────────────────────
            @testset "large array n=1024 N=4" begin
                n   = 1024
                src = AT(Float32.(1:n))
                dst = AT(zeros(Float32, n))
                KI.@kernel backend() numworkgroups = n ÷ 4 workgroupsize = 1 vload_copy_kernel(dst, src, Val(4))
                KernelAbstractions.synchronize(backend())
                @test Array(dst) == Array(src)
            end

            # ── 10. Multi-thread (workgroupsize > 1) ─────────────────────────
            @testset "multi-thread workgroupsize=4 N=4" begin
                wgs = 4
                n   = 128
                src = AT(Float32.(1:n))
                dst = AT(zeros(Float32, n))
                KI.@kernel backend() numworkgroups = n ÷ (wgs * 4) workgroupsize = wgs vload_mt_copy_kernel(dst, src, Val(4))
                KernelAbstractions.synchronize(backend())
                @test Array(dst) == Array(src)
            end

            # ── 11. Integer types ────────────────────────────────────────────
            @testset "Int32 N=4 arithmetic" begin
                n   = 64
                src = AT(Int32.(1:n))
                dst = AT(zeros(Int32, n))
                KI.@kernel backend() numworkgroups = n ÷ 4 workgroupsize = 1 vload_scale_kernel(dst, src, Int32(3), Val(4))
                KernelAbstractions.synchronize(backend())
                @test Array(dst) == 3 .* Int32.(1:n)
            end

            # ── 12. Type stability ───────────────────────────────────────────
            @testset "type stability T=$T N=$N" for T in vload_types, N in (1, 4)
                n   = N
                src = AT(T.(1:n))
                dst = AT(zeros(T, n))
                KI.@kernel backend() numworkgroups = 1 workgroupsize = 1 vload_copy_kernel(dst, src, Val(N))
                KernelAbstractions.synchronize(backend())
                @test eltype(Array(dst)) === T
            end

        end
    end
    return nothing
end
