# One work item per index of a collection.
#
# The index space is carried by the `ndrange`, so the kernels take no argument besides the
# function: `@index(Global, Cartesian)` already returns the index of the `ndrange`, including
# the offset of an index space that does not start at 1.
#
# `f` is inlined on purpose: an out-of-line call to a closure can spill its captures to local
# memory on some backends.

@kernel function foreach_index_linear_kernel(f)
    I = @index(Global, Cartesian)
    @inline f(I.I[1])
end

@kernel function foreach_index_cartesian_kernel(f)
    I = @index(Global, Cartesian)
    @inline f(I)
end

# `eachindex` of an `AbstractArray` is either a range of linear indices or a `CartesianIndices`;
# hand `f` the index type that `itr` is indexed with in either case.
foreach_index_kernel(::AbstractUnitRange, backend) = foreach_index_linear_kernel(backend)
foreach_index_kernel(::CartesianIndices, backend) = foreach_index_cartesian_kernel(backend)
foreach_index_kernel(indices, backend) = throw(
    ArgumentError(
        "`foreach_index` needs an index space that is a range of linear indices or a `CartesianIndices`, got `$(typeof(indices))`"
    )
)

"""
    foreach_index(f, itr, backend = get_backend(itr); workgroupsize = nothing)

Call `f(i)` once for every `i` in `eachindex(itr)`, with one work item per index.

This is the kernel-free spelling of a `for` loop over the indices of a collection: the body is
an ordinary Julia function, so the same code runs on every backend.

```julia
function scale!(y, x)
    foreach_index(x) do i
        @inbounds y[i] = 2 * x[i] + 1
    end
    return y
end
```

`f` receives the same index that a `for i in eachindex(itr)` loop would: a linear index for an
array with `IndexLinear` style, a `CartesianIndex` otherwise.

Like any other kernel launch this is asynchronous; call [`synchronize`](@ref) before reading the
result on the host. Bounds checks are not elided, so write `@inbounds` in the body where it is
warranted, as in a hand-written kernel. `workgroupsize` is passed on to the launch, and by
default the backend chooses it.

# Extended help

`f` becomes the body of a kernel, so it is subject to the same restrictions: every value it
captures must be of a known type. Closing over a variable of the enclosing *global* scope leaves
its type unknown and fails to compile (typically with `unsupported dynamic function invocation`),
which is why the example above wraps the loop in a function.

For the same reason `f` must not assign to a captured variable, as that makes Julia box the
capture; accumulate into a one-element array, with an atomic update if the indices race.

See also [`@kernel`](@ref) to write the kernel out, which is what to reach for when the body
needs more of the kernel language than an index (workgroup-level indices, local memory, or
synchronization).
"""
function foreach_index(f::F, itr, backend::Backend = get_backend(itr); workgroupsize = nothing) where {F}
    indices = eachindex(itr)
    isempty(indices) && return nothing
    kernel = foreach_index_kernel(indices, backend)
    kernel(f; ndrange = indices, workgroupsize)
    return nothing
end
