using KernelAbstractions
using oneAPI
using OneKernels
using Test

@testset "Testing OneKernels" begin

@kernel function vadd(a,b,c)
    I = @index(Global, Linear)
    J = @index(Local, Linear)
    K = @index(Group, Linear)
    c[I] = a[I] + b[I]
end


n = 10
a = ones(n)
b = ones(n)
c = zeros(n)

ev = vadd(CPU())(a,b,c, ndrange=10)
wait(ev)
fill!(c, 0.0)

da = oneArray(a)
db = oneArray(b)
dc = oneArray(c)

c .= a .+ b

fill!(c, 0.0)
dc = oneArray(c)
kernel = vadd(OneDevice())
ev = kernel(da,db,dc, ndrange=10)
wait(ev)
@test all(Array(dc) .== 2.0)

end
