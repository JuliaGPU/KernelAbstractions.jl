using KernelAbstractions
using Test

import KernelAbstractions.NDIteration: NDRange, StaticSize, DynamicSize

@kernel function index(A)
    I  = @index(Global, NTuple)
    @show A[I...]
end
kernel = index(CPU(), DynamicSize(), DynamicSize())
iterspace = NDRange{1, StaticSize{(128,)}, StaticSize{(8,)}}();
ctx = KernelAbstractions.mkcontext(kernel, 1, nothing, iterspace, Val(KernelAbstractions.NoDynamicCheck()))

@test KernelAbstractions.Cassette.overdub(ctx, KernelAbstractions.__index_Global_NTuple, CartesianIndex(1)) == (1,)