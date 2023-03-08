# Notes for backend implementations

## Semantics of `KernelAbstractions.synchronize`

[`KernelAbstractions.synchronize`](@ref) is required to be **cooperative**,
with that we mean it can not block inside an external library, but instead must
implement a cooperative wait that will `yield` the current task and return the
scheduling slice to the Julia runtime.

This is of particular import to allow for overlapping of communication and
computation with MPI.
