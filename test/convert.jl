using KernelAbstractions, Test

# Not passing in typelist because CuArrays will not support passing in
# non-inlined types
@kernel function convert_kernel!(A, B)
    tid = @index(Global, Linear)

    # Int -> Int
    tid = Int64(Int32(tid))

    @inbounds B[tid, 1] = ceil(Int8, A[tid])
    @inbounds B[tid, 2] = ceil(Int16, A[tid])
    @inbounds B[tid, 3] = ceil(Int32, A[tid])
    @inbounds B[tid, 4] = ceil(Int64, A[tid])
    @inbounds B[tid, 5] = ceil(Int128, A[tid])
    @inbounds B[tid, 6] = ceil(UInt8, A[tid])
    @inbounds B[tid, 7] = ceil(UInt16, A[tid])
    @inbounds B[tid, 8] = ceil(UInt32, A[tid])
    @inbounds B[tid, 9] = ceil(UInt64, A[tid])
    @inbounds B[tid, 10] = ceil(UInt128, A[tid])

    @inbounds B[tid, 11] = floor(Int8, A[tid])
    @inbounds B[tid, 12] = floor(Int16, A[tid])
    @inbounds B[tid, 13] = floor(Int32, A[tid])
    @inbounds B[tid, 14] = floor(Int64, A[tid])
    @inbounds B[tid, 15] = floor(Int128, A[tid])
    @inbounds B[tid, 16] = floor(UInt8, A[tid])
    @inbounds B[tid, 17] = floor(UInt16, A[tid])
    @inbounds B[tid, 18] = floor(UInt32, A[tid])
    @inbounds B[tid, 19] = floor(UInt64, A[tid])
    @inbounds B[tid, 20] = floor(UInt128, A[tid])

    @inbounds B[tid, 21] = round(Int8, A[tid])
    @inbounds B[tid, 22] = round(Int16, A[tid])
    @inbounds B[tid, 23] = round(Int32, A[tid])
    @inbounds B[tid, 24] = round(Int64, A[tid])
    @inbounds B[tid, 25] = round(Int128, A[tid])
    @inbounds B[tid, 26] = round(UInt8, A[tid])
    @inbounds B[tid, 27] = round(UInt16, A[tid])
    @inbounds B[tid, 28] = round(UInt32, A[tid])
    @inbounds B[tid, 29] = round(UInt64, A[tid])
    @inbounds B[tid, 30] = round(UInt128, A[tid])

end

function convert_testsuite(backend, ArrayT)
    ET = KernelAbstractions.supports_float64(backend()) ? Float64 : Float32

    N = 32
    d_A = ArrayT([rand(ET) * 3 for i in 1:N])

    # 30 because we have 10 integer types and we have 3 operations
    d_B = ArrayT(zeros(ET, N, 30))

    @testset "convert test" begin
        kernel = convert_kernel!(backend(), 4)
        kernel(d_A, d_B, ndrange = (N))
        synchronize(backend())

        for i in 1:10
            @test d_B[:, i] == ceil.(d_A)
            @test d_B[:, i + 10] == floor.(d_A)
            @test d_B[:, i + 20] == round.(d_A)
        end
    end
    return
end
