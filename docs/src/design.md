# Design notes 

- Loops are affine
- Operation over workgroups/blocks
- Goal: Kernel fusion
- `@Const`:
    - `restrict const` in C
    - `ldg` on the GPU
    - `@aliasscopes` on the CPU

- Cartesian or Linear indicies supported
  - `@index(Linear)
  - `@index(Cartesian)
- `@synchronize` for inserting workgroup-level synchronization
- workgroupsize constant
  - may allow for `Dynamic()`
- terminology -- how much to borrow from OpenCL
- http://portablecl.org/docs/html/kernel_compiler.html#work-group-function-generation

## TODO
- Do we want to support Cartesian indices?
  - Just got removed from GPUArrays
  - recovery is costly
  - Going from Cartesian to linear sometimes confuses LLVM (IIRC this is true for dynamic strides, due to overflow issues)
- `@index(Global, Linear)`
- Support non-multiple of workgroupsize
  - do we require index inbounds checks?
    - Harmful for CPU vectorization -- likely want to generate two kernels
- Multithreading requires 1.3
- Tests
- Docs
- Examples
- Index calculations
- inbounds checks on the GPU
- 
