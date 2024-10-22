using KernelAbstractions, Test
include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend

@kernel function copy_kernel!(A, @Const(B))
    I = @index(Global)
    @inbounds A[I] = B[I]
end

function mycopy!(A, B)
    backend = get_backend(A)
    @assert size(A) == size(B)
    @assert get_backend(B) == backend

    kernel = copy_kernel!(backend)
    kernel(A, B, ndrange = length(A))
    return
end

A = KernelAbstractions.zeros(backend, Float64, 128, 128)
B = KernelAbstractions.ones(backend, Float64, 128, 128)
mycopy!(A, B)
KernelAbstractions.synchronize(backend)
@test A == B
