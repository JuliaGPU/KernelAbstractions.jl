using KernelAbstractions, Test

@kernel function unsafe_getindex_kernel!(A, @Const(B))
    I = @index(Global, Linear)
    @inbounds A[I] = Base._unsafe_getindex(IndexStyle(B), B, I)[1]
end

function unsafe_getindex(backend, ArrayT)
    A = convert(ArrayT, zeros(128))
    B = convert(ArrayT, ones(128))
    kernel = unsafe_getindex_kernel!(backend(), 8)
    ev = kernel(A, B, ndrange=length(A))
    wait(ev)
    @test A == B
end
