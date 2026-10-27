# The time to launch a kernel that does next to nothing, until it has completed: the host
# side of a launch, and handing the work to the backend. Each benchmark takes a different
# path through `src/backend_launch.jl`. The first two don't allocate (`runbenchmarks.jl`
# checks that). Launches with a static workgroup size, on one worker, or with a 4-dimensional
# range varied too much between runs to be tracked; the last is covered by
# `performance/.../indexing/ntuple4d`.

@kernel function launch_affine!(y, @Const(x))
    i = @index(Global, Linear)
    @inbounds y[i] = 2.0f0 * x[i] + 1.0f0
end

# 39 scalar arguments, all of which contribute to the result
const LAUNCH_NARGS = 39
let args = [Symbol(:a, i) for i in 1:LAUNCH_NARGS]
    @eval @kernel function launch_many_args!(y, @Const(x), $(args...))
        i = @index(Global, Linear)
        @inbounds y[i] = x[i] + $(foldl((a, b) -> :($a + $b), args))
    end
end

@kernel function launch_groupreduce!(out, @Const(x))
    i = @index(Global, Linear)
    s = @groupreduce(+, x[i], zero(eltype(out)))
    if @index(Local, Linear) == 1
        @inbounds out[@index(Group, Linear)] = s
    end
end

function launch_data(T, dims...)
    x = KernelAbstractions.allocate(BACKEND, T, dims...)
    copyto!(x, reshape(T.(1:prod(dims)), dims))
    return x, KernelAbstractions.zeros(BACKEND, T, dims...)
end

# The workgroup size is chosen at launch
bench("latency/pocl/parallel/launch/dynamic-default") do
    x, y = launch_data(Float32, 64)
    kernel = launch_affine!(BACKEND)
    f() = (kernel(y, x; ndrange = 64); synchronize(BACKEND))
    f()
    @assert y == 2 .* x .+ 1
    measure(f, LAUNCH)
end

# The last workgroup is partial, so the kernel checks the bounds of its work-items
bench("latency/pocl/parallel/launch/partial-workgroup") do
    x, y = launch_data(Float32, 65)
    kernel = launch_affine!(BACKEND, 64)
    f() = (kernel(y, x; ndrange = 65); synchronize(BACKEND))
    f()
    @assert y == 2 .* x .+ 1
    measure(f, LAUNCH)
end

# Converting and passing many arguments
bench("latency/pocl/parallel/launch/many-args") do
    x, y = launch_data(Float32, 64)
    args = ntuple(Float32, LAUNCH_NARGS)
    kernel = launch_many_args!(BACKEND, 64)
    f() = (kernel(y, x, args...; ndrange = 64); synchronize(BACKEND))
    f()
    @assert y == x .+ sum(args)
    measure(f, LAUNCH)
end

# A kernel with a collective, for which the launch queries what the device supports
bench("latency/pocl/parallel/launch/groupreduce-f64") do
    x, _ = launch_data(Float64, 64)
    out = KernelAbstractions.zeros(BACKEND, Float64, 1)
    kernel = launch_groupreduce!(BACKEND, 64)
    f() = (kernel(out, x; ndrange = 64); synchronize(BACKEND))
    f()
    @assert out[1] == sum(x)
    measure(f, LAUNCH)
end
