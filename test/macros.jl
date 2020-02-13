using KernelAbstractions
using Test

@kernel function nodynamic(A)
    I = @index(Global, Linear)
    A[I] = I
end

@kernel function dynamic(A)
    I = @index(Global, Linear)
    i = @index(Local, Linear)
    lmem = @dynamic_localmem eltype(A) (wkrgpsize)->2*wkrgpsize
    lmem[2*i] = A[I] 
    @synchronize
    A[I] = lmem[2*i]
end

nodyn_kernel = nodynamic(CPU())
dyn_kernel   = dynamic(CPU())

@test KernelAbstractions.allocator(nodyn_kernel.f)(zeros(32, 32)) == ()
let allocs = KernelAbstractions.allocator(dyn_kernel.f)(zeros(32, 32))
    bytes, alloc = allocs[1]
    @test bytes == sizeof(Float64)
    @test alloc(32) == 64
end
