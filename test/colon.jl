using KernelAbstractions, Test

@kernel function copy_colon_kernel!(A, @Const(B))
    I = @index(Global)
    @inbounds A[I] = sum(B[:,I])
end

function copy_colon(backend, ArrayT)
    A = convert(ArrayT, zeros(128))
    B = convert(ArrayT, ones(2,128))
    kernel = copy_colon_kernel!(backend(), 8)
    ev = kernel(A, B, ndrange=length(A))
    wait(ev)
    @test all(A .== 2.0)
end
