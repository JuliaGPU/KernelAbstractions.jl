# Writing kernels 

These kernel language constructs are intended to be used as part
of [`@kernel`](@ref) functions and not valid outside that context.

## Constant arguments

Kernel functions allow for input arguments to be marked with the
[`@Const`](@ref) macro. It informs the compiler that the memory
accessed through that marked input argument, will not be written
to as part of the kernel. This has the implication that input arguments
are **not** allowed to alias each other. If you are used to CUDA C this
is similar to `const restrict`.

## Indexing

There are several [`@index`](@ref) variants.

## Local memory, variable lifetime and private memory

[`@localmem`](@ref), [`@synchronize`](@ref), [`@private`](@ref)

# Launching kernels

## [Kernel dependencies](@id dependencies)
