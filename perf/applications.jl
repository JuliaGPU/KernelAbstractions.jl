# Larger kernels, and sequences of them, taken from applications

include("dgsem.jl")


## DGSEM volume integrals, Float64 with 4 nodes per direction (see dgsem.jl)

const DGSEM_N = 4

# The weak-form variants compute the same `du`, so the one per element (which is the
# simplest) checks the other
function dgsem_bench(f!, ::Val{DIM}, nelem; reference = nothing) where {DIM}
    data = dgsem_data(BACKEND, Float64, Val(DIM), DGSEM_N, nelem)
    du = KernelAbstractions.zeros(BACKEND, Float64, size(data.u)...)
    f() = (f!(du, data); synchronize(BACKEND))
    f()
    result = Array(du)
    all(isfinite, result) && count(iszero, result) < length(result) ÷ 100 &&
        (reference === nothing || result ≈ reference(data)) || error("wrong result")
    return measure(f, APPLICATION)
end

function dgsem_weak_reference(data)
    nelem = size(data.u)[end]
    du = KernelAbstractions.zeros(BACKEND, Float64, size(data.u)...)
    dgsem_weak_element!(BACKEND, 64)(du, data.u, data.D, data.Ja, Val(DGSEM_N), Val(3); ndrange = nelem)
    synchronize(BACKEND)
    return Array(du)
end

bench("applications/pocl/parallel/dgsem/weak-element") do
    nelem = 8^3
    kernel = dgsem_weak_element!(BACKEND, 64)
    dgsem_bench(Val(3), nelem) do du, data
        kernel(du, data.u, data.D, data.Ja, Val(DGSEM_N), Val(3); ndrange = nelem)
    end
end

bench("applications/pocl/parallel/dgsem/weak-node-localmem") do
    nelem, E = 8^3, 2
    nodes = ntuple(_ -> DGSEM_N, 3)
    kernel = dgsem_weak_node_localmem!(BACKEND, (nodes..., E))
    layout = dgsem_default_layout(BACKEND)
    dgsem_bench(Val(3), nelem; reference = dgsem_weak_reference) do du, data
        kernel(du, data.u, data.D, data.Ja, Val(DGSEM_N), Val(3), Val(E), layout; ndrange = (nodes..., nelem))
    end
end

bench("applications/pocl/parallel/dgsem/fluxdiff-node") do
    nelem, E = 32^2, 8
    nodes = ntuple(_ -> DGSEM_N, 2)
    kernel = dgsem_fluxdiff_node!(BACKEND, (nodes..., E))
    dgsem_bench(Val(2), nelem) do du, data
        kernel(du, data.u, data.Ds, data.Ja, Val(DGSEM_N), Val(2); ndrange = (nodes..., nelem))
    end
end


## a sequence of 7-point Jacobi updates of a 128³ grid with a fixed boundary, alternating
## between two arrays: many launches of a short, memory-bound kernel (#906)

@kernel function jacobi7!(b, @Const(a))
    i, j, k = @index(Global, NTuple)
    i, j, k = i + 1, j + 1, k + 1
    @inbounds b[i, j, k] = (
        a[i - 1, j, k] + a[i + 1, j, k] + a[i, j - 1, k] +
            a[i, j + 1, k] + a[i, j, k - 1] + a[i, j, k + 1]
    ) * (1.0f0 / 6)
end

const STENCIL_SIZE = 128
const STENCIL_STEPS = 10

for mode in ("parallel", "serial")
    bench("applications/pocl/$mode/stencil/timestep") do
        n = STENCIL_SIZE
        initial = rand(Random.Xoshiro(42), Float32, n + 2, n + 2, n + 2)
        a = copyto!(KernelAbstractions.allocate(BACKEND, Float32, size(initial)), initial)
        b = copyto!(similar(a), initial)
        kernel = jacobi7!(BACKEND, (32, 4, 2))
        function f()
            for _ in 1:(STENCIL_STEPS ÷ 2)
                kernel(b, a; ndrange = (n, n, n))
                kernel(a, b; ndrange = (n, n, n))
            end
            return synchronize(BACKEND)
        end
        reset() = copyto!(a, initial)
        reset()
        f()
        result, interior = Array(a), 2:(n + 1)
        all(isfinite, result) && result[1, :, :] == initial[1, :, :] &&
            result[interior, interior, interior] != initial[interior, interior, interior] ||
            error("wrong result")
        measure(f, APPLICATION; setup = reset)
    end
end
