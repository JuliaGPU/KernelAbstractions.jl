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

# A zero-dimensional index space has a single index. Launched as a one-dimensional `ndrange`, as
# a workgroup size has at least one dimension.
@kernel function foreach_index_zerodim_kernel(f)
    @inline f(CartesianIndex())
end

# The index spaces an `ndrange` can express: a range of integers, or a product of them.
const UnitCartesianIndices{N} = CartesianIndices{N, <:NTuple{N, AbstractUnitRange{Int}}}

foreach_index_kernel(backend, ::AbstractUnitRange{<:Integer}) = foreach_index_linear_kernel(backend)
foreach_index_kernel(backend, ::UnitCartesianIndices) = foreach_index_cartesian_kernel(backend)
foreach_index_kernel(backend, ::CartesianIndices{0}) = foreach_index_zerodim_kernel(backend)
foreach_index_kernel(backend, indices) = throw(
    ArgumentError(
        "`foreach_index` needs an index space that is a range of integers or a `CartesianIndices` of such ranges, got a `$(typeof(indices))`"
    )
)

# A captured variable that is assigned to after the closure was created, or in it, is stored in
# a `Core.Box`, which a kernel cannot access. That fails to compile with an error about the
# kernel's arguments, so catch it here with one about the variable.
@inline function check_captures(f::F) where {F}
    has_box(fieldtypes(F)) && boxed_capture_error(f)
    return
end

# (`any` would not be folded by Julia 1.10)
has_box(::Tuple{}) = false
has_box(Ts::Tuple) = first(Ts) === Core.Box || has_box(Base.tail(Ts))

@noinline function boxed_capture_error(::F) where {F}
    names = [fieldname(F, i) for i in 1:fieldcount(F) if fieldtype(F, i) === Core.Box]
    vars = join(("`$name`" for name in names), ", ", " and ")
    throw(
        ArgumentError(
            "`foreach_index` cannot run a function that captures a variable that is reassigned " *
                "($vars): Julia stores such a variable in a box, which a kernel cannot access. " *
                "Capture a variable that is not reassigned instead, e.g. by wrapping the loop in " *
                "`let $(first(names)) = $(first(names))`, and write results to an array."
        )
    )
end

foreach_index_ndrange(indices) = indices
foreach_index_ndrange(::CartesianIndices{0}) = 1

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
every array that the body indexes with `i`, so that the index is valid for each of them. Ranges,
`CartesianIndices` and `LinearIndices`, and views or reshapes of them, have no backend and run
on that of the other arrays; if none of the arrays has a backend, use the second form.

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

For the same reason `f` cannot capture a variable that is assigned to after `f` is created, or
in `f`, as Julia then boxes the variable. Bind the value to a new variable (e.g. with `let`)
for `f` to capture, and accumulate results into an array, with an atomic update if the indices
race.

On the `CPU` backend `foreach_index` also launches a kernel, compiled for every new `f`. For a
loop that runs once, or over few indices, a threaded loop (`Threads.@threads`) is cheaper.

See also [`@kernel`](@ref) to write the kernel out, which is what to reach for when the body
needs more of the kernel language than an index (workgroup-level indices, local memory, or
synchronization).
"""
function foreach_index(f::F, backend::Backend, indices; workgroupsize = nothing) where {F}
    kernel = foreach_index_kernel(backend, indices)
    check_captures(f)
    isempty(indices) && return nothing
    kernel(f; ndrange = foreach_index_ndrange(indices), workgroupsize)
    return nothing
end

# The backend of the arrays passed to `foreach_index`. Ranges, `CartesianIndices` and
# `LinearIndices` are computed rather than stored, as are Base's views, reshapes and permutations
# of them: they have no backend and do not decide the one the loop runs on. This is what
# AcceleratedKernels.jl does.
array_backend(A::AbstractArray) = get_backend(A)
array_backend(::Union{AbstractRange, CartesianIndices, LinearIndices}) = nothing
array_backend(A::Union{SubArray, Base.ReshapedArray, PermutedDimsArray}) = array_backend(parent(A))

common_backend() = nothing
common_backend(A, Bs...) = merge_backend(array_backend(A), common_backend(Bs...))

merge_backend(::Nothing, ::Nothing) = nothing
merge_backend(a, ::Nothing) = a
merge_backend(::Nothing, b) = b
function merge_backend(a, b)
    a == b || throw(ArgumentError("`foreach_index` needs arrays with the same backend, got arrays on $a and on $b"))
    return a
end

function foreach_index(f::F, A::AbstractArray, Bs::AbstractArray...; workgroupsize = nothing) where {F}
    backend = common_backend(A, Bs...)
    backend === nothing && throw(
        ArgumentError(
            "`foreach_index` cannot determine a backend from arrays that are not stored, such as ranges; pass it explicitly, as in `foreach_index(f, backend, eachindex(A, Bs...))`"
        )
    )
    return foreach_index(f, backend, eachindex(A, Bs...); workgroupsize)
end
