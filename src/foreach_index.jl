# One work item per index of an index space.
#
# The index space is carried by the `ndrange`, so the kernels take no argument besides the
# function: `@index(Global, Cartesian)` already returns the index of the `ndrange`, including
# the offset of an index space that does not start at 1.
#
# `f` is inlined on purpose: an out-of-line call to a closure can spill its captures to local
# memory on some backends.

@kernel function foreach_index_linear_kernel(f)
    I = @index(Global, Cartesian)
    @inline f(I[1])
end

@kernel function foreach_index_cartesian_kernel(f)
    I = @index(Global, Cartesian)
    @inline f(I)
end

# The index spaces an `ndrange` can express: a range of integers, or a product of them.
const UnitCartesianIndices{N} = CartesianIndices{N, <:NTuple{N, AbstractUnitRange{Int}}}

foreach_index_kernel(backend, ::AbstractUnitRange{<:Integer}) = foreach_index_linear_kernel(backend)
foreach_index_kernel(backend, ::UnitCartesianIndices) = foreach_index_cartesian_kernel(backend)
foreach_index_kernel(backend, indices) = throw(
    ArgumentError(
        "`foreach_index` needs an index space that is a range of integers or a `CartesianIndices` of such ranges, got a `$(typeof(indices))`"
    )
)

"""
    foreach_index(f, A::AbstractArray, Bs::AbstractArray...)
    foreach_index(f, backend::Backend, indices)

Call `f(i)` once for every index `i` of the arrays `A, Bs...`, or for every index `i` in `indices`,
with one work item per index. Returns `nothing`; the iterations run asynchronously.

This is a `for` loop over indices without a kernel to write out: the body is an ordinary Julia
function, which becomes the body of a kernel.

```julia
function scale!(y, x)
    foreach_index(y, x) do i
        @inbounds y[i] = 2 * x[i] + 1
    end
    return y
end

scale!(y, x)
synchronize(get_backend(y))
```

The first form runs on the backend of the arrays, which must all have the same one, over
`eachindex(A, Bs...)`: `f` receives the index that a `for i in eachindex(A, Bs...)` loop would,
a linear index if the arrays have `IndexLinear` style and a `CartesianIndex` otherwise. Pass
every array that the body indexes with `i`, so that the index is valid for each of them.

The second form runs on `backend`, over the given `indices`: a range of integers such as
`1:n` or `axes(A, 2)`, for which `f` receives an `Int`, or a `CartesianIndices` of such ranges,
for which `f` receives a `CartesianIndex`. Indices that do not start at 1 are passed on as
they are, so this iterates, e.g., the interior of a 2-D array `A`:

```julia
foreach_index(get_backend(A), CartesianIndices((2:size(A, 1)-1, 2:size(A, 2)-1))) do I
    ...
end
```

The iterations run concurrently and in no particular order, so they must not race on the
same memory. The value `f` returns is ignored. Like any other kernel launch, `foreach_index`
returns before the iterations have finished: call [`synchronize`](@ref) before reading their
results on the host. Bounds checks are not elided, so write `@inbounds` in the body where it is
warranted, as in a hand-written kernel.

The keyword argument `workgroupsize` sets the workgroup size of the launch; by default the
backend chooses it.

# Extended help

`f` is subject to the same restrictions as a kernel: every value it captures must be of a known
type. Closing over a variable of the enclosing *global* scope leaves its type unknown and fails
to compile (typically with `unsupported dynamic function invocation`), which is why the example
above wraps the loop in a function.

For the same reason `f` must not assign to a captured variable, as that makes Julia box the
capture; accumulate into a one-element array, with an atomic update if the indices race.

On the `CPU` backend `foreach_index` also launches a kernel, compiled for every new `f`. For a
loop that runs once, or over few indices, a threaded loop (`Threads.@threads`) is cheaper.

See also [`@kernel`](@ref) to write the kernel out, which is what to reach for when the body
needs more of the kernel language than an index (workgroup-level indices, local memory, or
synchronization).
"""
function foreach_index(f::F, backend::Backend, indices; workgroupsize = nothing) where {F}
    kernel = foreach_index_kernel(backend, indices)
    isempty(indices) && return nothing
    kernel(f; ndrange = indices, workgroupsize)
    return nothing
end

function foreach_index(f::F, A::AbstractArray, Bs::AbstractArray...; workgroupsize = nothing) where {F}
    backend = get_backend(A)
    for B in Bs
        get_backend(B) == backend || throw(
            ArgumentError(
                "`foreach_index` needs arrays with the same backend, got a `$(typeof(A))` on $(backend) and a `$(typeof(B))` on $(get_backend(B))"
            )
        )
    end
    return foreach_index(f, backend, eachindex(A, Bs...); workgroupsize)
end
