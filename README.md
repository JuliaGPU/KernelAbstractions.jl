KernelAbstractions.jl
==============
KernelAbstractions (KA) is
a package that enables you to write GPU-like kernels targetting different
execution backends. KA is intended to be a minimal and
performant library that explores ways to write heterogeneous code.
Currently, the following backends are supported:

* [NVIDIA CUDA](https://github.com/JuliaGPU/CUDA.jl)
* [AMD ROCm](https://github.com/JuliaGPU/AMDGPU.jl)
* [Intel oneAPI](https://github.com/JuliaGPU/oneAPI.jl)
* [Apple Metal](https://github.com/JuliaGPU/Metal.jl)

[![Documentation (stable)][docs-stable-img]][docs-stable-url]
[![Documentation (latest)][docs-latest-img]][docs-latest-url]
[![DOI][doi-img]][doi-url]
[![Code Coverage][codecov-img]][codecov-url]

| CPU CI                                                             | GPU CI                                                                    |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------  |
| [![CI][ci-img]][ci-url]                                            | [![Build status][buildkite-img]][buildkite-url]                    |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://juliagpu.github.io/KernelAbstractions.jl/stable
[docs-latest-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-latest-url]: https://juliagpu.github.io/KernelAbstractions.jl/dev
[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.4021259.svg
[doi-url]: https://doi.org/10.5281/zenodo.4021259
[codecov-img]: https://codecov.io/gh/JuliaGPU/KernelAbstractions.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaGPU/KernelAbstractions.jl
[ci-img]: https://github.com/JuliaGPU/KernelAbstractions.jl/actions/workflows/ci.yml/badge.svg?branch=main
[ci-url]: https://github.com/JuliaGPU/KernelAbstractions.jl/actions/workflows/ci.yml?query=workflow%3ACI
[buildkite-img]: https://badge.buildkite.com/1509baa1122772e8ec377463a6c188753d35b8fcec300a658e.svg?branch=main
[buildkite-url]: https://buildkite.com/julialang/kernelabstractions-dot-jl

License
-------

KernelAbstractions.jl is licensed under the [MIT license](LICENSE.md).

Cite this package as
--------------------

```
@software{Churavy_KernelAbstractions_jl,
author = {Churavy, Valentin},
license = {MIT},
title = {{KernelAbstractions.jl}},
url = {https://github.com/JuliaGPU/KernelAbstractions.jl}
doi = {10.5281/zenodo.4021259},
}
```
