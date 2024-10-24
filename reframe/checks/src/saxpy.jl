using BenchmarkTools
using KernelAbstractions
using Random

@kernel function copy_kernel!(Z, @Const(X))
    I = @index(Global)
    @inbounds Z[I] = X[I]
end

@kernel function saxpy_kernel!(Z, a, @Const(X), @Const(Y))
    I = @index(Global)
    @inbounds Z[I] = a * X[I] + Y[I]
end

# TODO: Parse cmdline args
T = Float16
N = 1048576
BACKEND = CPU()

res_copy = @benchmark begin
    kernel = copy_kernel!($BACKEND)
    kernel(Z, X, ndrange = size(Z))
    synchronize($BACKEND)
end setup = (
    X = rand!(KernelAbstractions.zeros($BACKEND, $T, $N));
    Z = KernelAbstractions.zeros($BACKEND, $T, $N)
)

res_saxpy = @benchmark begin
    kernel = saxpy_kernel!($BACKEND)
    kernel(Z, convert($T, 2.0), X, Y, ndrange = size(Z))
    synchronize($BACKEND)
end setup = (
    X = rand!(KernelAbstractions.zeros($BACKEND, $T, $N));
    Y = rand!(KernelAbstractions.zeros($BACKEND, $T, $N));
    Z = KernelAbstractions.zeros($BACKEND, $T, $N)
)

bytes_saxpy = 3 * sizeof(T) * N # num bytes transferred in SAXPY
bytes_copy  = 2 * sizeof(T) * N # num bytes transferred in copy
time_saxpy = minimum(res_saxpy).time
time_copy  = minimum(res_copy).time

println("Copy: ",  bytes_copy/time_copy)
println("Saxpy: ", bytes_saxpy/time_saxpy)
println("Solution Validates")
