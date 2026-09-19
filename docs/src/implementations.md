# [Notes for backend implementations](@id implementations_notes)

The [KernelInterface](@ref kernelinterface) sibling package defines the core interface a backend must implement. A backend must implement a backend type that subtypes `KernelInterface.GPU`, or `KernelInterface.Backend` for non-gpu backends. This documentation contains the host and devices side functions that backends can define, as well as whether they are mandatory or not.

## Semantics of `KernelAbstractions.synchronize`

[`KernelAbstractions.synchronize`](@ref) is required to be **cooperative**,
with that we mean it can not block inside an external library, but instead must
implement a cooperative wait that will `yield` the current task and return the
scheduling slice to the Julia runtime.

This is of particular import to allow for overlapping of communication and
computation with MPI.

## Moving data with `adapt`

KernelAbstractions extends [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl) so
that [`adapt(backend, x)`](@ref Adapt.adapt_storage(::Backend, ::Any)) moves the
arrays in `x` to `backend`, without the caller having to know the backend's
array type. Every backend **must** support this by extending
`Adapt.adapt_storage` for its backend type. The recommended definition delegates
to the backend's array type, so that `adapt(backend, x)` and
`adapt(BackendArray, x)` agree:

```julia
Adapt.adapt_storage(::CUDABackend, x) = adapt(CuArray, x)
```

## Iteration spaces and the validity of work items

The context a backend passes to a kernel carries the blocked iteration space,
[`__iterspace(ctx)`](@ref KernelAbstractions.NDIteration.NDRange), and the
`ndrange` of the launch, `__ndrange(ctx)`. A backend must derive everything
about a work item from these two objects through three functions:

- [`expand(iterspace, groupidx, idx)`](@ref KernelAbstractions.NDIteration.expand)
  gives the `CartesianIndex` handled by work item `idx` of workgroup `groupidx`;
- `expand(iterspace, groupidx, idx) in __ndrange(ctx)` tells whether that work
  item has an index to handle, which is how `__validindex` must be implemented;
- [`linear_index(__ndrange(ctx), I)`](@ref KernelAbstractions.NDIteration.linear_index)
  gives the linear index of `I`.

A backend must not assume that `__ndrange(ctx)` is a `CartesianIndices` or that
`expand` is an affine map: the `mapping` field of the `NDRange` lets a package
define its own iteration space, for example a list of indices to visit, by
extending these functions for its mapping type. Overriding `__validindex` or
`__index_Global_Linear` for a generic `ctx` would bypass such an extension.
See [`NDRange`](@ref KernelAbstractions.NDIteration.NDRange) for the functions
a custom mapping has to define.

