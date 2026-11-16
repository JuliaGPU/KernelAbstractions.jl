# The time to launch the first kernel of a process, until it has completed. This includes
# compiling it, and setting up the backend.

using KernelAbstractions

@kernel function first_kernel!(y, @Const(x))
    i = @index(Global, Linear)
    @inbounds y[i] = 2.0f0 * x[i] + 1.0f0
end

const backend = CPU()
x = copyto!(KernelAbstractions.allocate(backend, Float32, 256), Float32.(1:256))
y = KernelAbstractions.zeros(backend, Float32, 256)
kernel = first_kernel!(backend, 256)
t = @elapsed begin
    kernel(y, x; ndrange = 256)
    synchronize(backend)
end
y == 2 .* x .+ 1 || error("wrong result")
println(t)
