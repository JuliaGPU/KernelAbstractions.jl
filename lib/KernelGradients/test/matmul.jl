# Simple kernel for matrix multiplication
@kernel function matmul_kernel!(a, b, c)
    i, j = @index(Global, NTuple)

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = zero(eltype(c))
    for k = 1:size(a)[2]
        @inbounds tmp_sum += a[i,k] * b[k, j]
    end

    c[i,j] = tmp_sum
end


function matmul_testsuite(backend, ArrayT)

    matmul = matmul_kernel!(backend(), (32, 32))
    a = ArrayT(rand(128, 256))
    b = ArrayT(rand(256, 128))
    c = ArrayT(zeros(128, 128))
    wait(matmul(a, b, c, ndrange=size(c)))

    @test c ≈ a*b

    dmatmul = Enzyme.autodiff(matmul)
    da = similar(a)
    da .= 0
    db = similar(b)
    db .= 0
    dc = similar(c)
    dc .= 1
    c .= 0

    compare_dc = copy(dc)
    wait(dmatmul(
        Duplicated(a, da),
        Duplicated(b, db),
        Duplicated(c, dc), ndrange=size(c)))

    @test da ≈ compare_dc * b'
    @test db ≈ a' * compare_dc
end