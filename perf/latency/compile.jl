# The time to launch a kernel that hasn't been compiled yet, until it has completed, in a
# process that has already launched another kernel:
#
#     julia compile.jl CASE
#
# - `new-kernel`: a simple kernel;
# - `localmem-barrier`: a kernel exchanging values through local memory;
# - `groupreduce`: a kernel with a work-group reduction;
# - `new-workgroupsize`: a kernel that has been launched, with another workgroup size.

using KernelAbstractions

@kernel function warmup!(y, @Const(x))
    i = @index(Global, Linear)
    @inbounds y[i] = x[i] - 1.0f0
end

@kernel function affine!(y, @Const(x))
    i = @index(Global, Linear)
    @inbounds y[i] = 2.0f0 * x[i] + 1.0f0
end

@kernel function exchange!(y, @Const(x))
    i = @index(Global, Linear)
    li = @index(Local, Linear)
    buf = @localmem Float32 (256,)
    @inbounds buf[li] = x[i]
    @synchronize
    @inbounds y[i] = buf[mod1(li + 1, 256)]
end

@kernel function groupsum!(out, @Const(x))
    i = @index(Global, Linear)
    s = @groupreduce(+, x[i], zero(eltype(out)))
    if @index(Local, Linear) == 1
        @inbounds out[@index(Group, Linear)] = s
    end
end

const backend = CPU()
const case = ARGS[1]
compilations() = KernelAbstractions.POCL.compilations[]

x = copyto!(KernelAbstractions.allocate(backend, Float32, 4096), Float32.(1:4096))
y = KernelAbstractions.zeros(backend, Float32, 4096)
warmup!(backend, 256)(y, x; ndrange = length(x))
synchronize(backend)

if case == "new-kernel"
    kernel = affine!(backend, 256)
    f() = kernel(y, x; ndrange = 256)
    check() = y[1:256] == 2 .* x[1:256] .+ 1
elseif case == "localmem-barrier"
    kernel = exchange!(backend, 256)
    f() = kernel(y, x; ndrange = 256)
    check() = y[1:256] == circshift(x[1:256], -1)
elseif case == "groupreduce"
    x64 = Float64.(Array(x)[1:256])
    out = KernelAbstractions.zeros(backend, Float64, 1)
    kernel = groupsum!(backend, 256)
    f() = kernel(out, x64; ndrange = 256)
    check() = Array(out)[1] == sum(x64)
elseif case == "new-workgroupsize"
    kernel = affine!(backend)
    kernel(y, x; ndrange = length(x), workgroupsize = 128)
    synchronize(backend)
    fill!(y, 0)
    f() = kernel(y, x; ndrange = length(x), workgroupsize = 256)
    check() = y == 2 .* x .+ 1
else
    error("Unknown case $case")
end

before = compilations()
t = @elapsed begin
    f()
    synchronize(backend)
end
check() || error("wrong result")
# a new workgroup size may only be compiled by PoCL
case == "new-workgroupsize" || compilations() > before || error("$case didn't compile a kernel")
println(t)
