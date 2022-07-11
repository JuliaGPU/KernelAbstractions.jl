KernelAbstractions.jl
==============
KernelAbstractions (KA) is
a package that enables you to write GPU-like kernels targetting different
execution [backends](backends.md). KA is intended to be a minimal and
performant library that explores ways to write heterogeneous code.
Currently, the following backends are supported:

* [NVIDIA CUDA](https://github.com/JuliaGPU/CUDA.jl),
* [AMD ROCm](https://github.com/JuliaGPU/AMDGPU.jl),
* [Intel oneAPI](https://github.com/JuliaGPU/oneAPI.jl).

[![Documentation (stable)][docs-stable-img]][docs-stable-url]
[![Documentation (latest)][docs-latest-img]][docs-latest-url]
[![DOI][doi-img]][doi-url]
[![Code Coverage][codecov-img]][codecov-url]

| Julia       | CPU CI                                                             | GPU CI                                                                    |
| ----------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------  |
| v1          | [![CI][ci-img]][ci-url]                                            | [![Build status][buildkite-julia1-img]][buildkite-url]                    |
| 1.6-nightly | [![CI][ci-julia-1-6-nightly-img]][ci-julia-1-6-nightly-url]        | [![Build status][buildkite-julia1.6nightly-img]][buildkite-url]           |
| nightly     | [![CI][ci-julia-nightly-img]][ci-julia-nightly-url]                | [![Build status][buildkite-julianightly-img]][buildkite-url]              |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://juliagpu.github.io/KernelAbstractions.jl/stable
[docs-latest-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-latest-url]: https://juliagpu.github.io/KernelAbstractions.jl/dev
[doi-img]: https://zenodo.org/badge/237471203.svg
[doi-url]: https://zenodo.org/badge/latestdoi/237471203
[codecov-img]: https://codecov.io/gh/JuliaGPU/KernelAbstractions.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaGPU/KernelAbstractions.jl
[ci-img]: https://github.com/JuliaGPU/KernelAbstractions.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/JuliaGPU/KernelAbstractions.jl/actions?query=workflow%3ACI
[ci-julia-1-6-nightly-img]: https://github.com/JuliaGPU/KernelAbstractions.jl/workflows/CI%20(Julia%201.6-nightly)/badge.svg
[ci-julia-1-6-nightly-url]: https://github.com/JuliaGPU/KernelAbstractions.jl/actions?query=workflow%3A%22CI+%28Julia+1.6-nightly%29%22
[ci-julia-nightly-img]: https://github.com/JuliaGPU/KernelAbstractions.jl/workflows/CI%20(Julia%20nightly)/badge.svg
[ci-julia-nightly-url]: https://github.com/JuliaGPU/KernelAbstractions.jl/actions?query=workflow%3A%22CI+%28Julia+nightly%29%22
[buildkite-julia1-img]: https://badge.buildkite.com/1509baa1122772e8ec377463a6c188753d35b8fcec300a658e.svg?branch=master&step=Julia%20v1
[buildkite-julia1.6nightly-img]: https://badge.buildkite.com/1509baa1122772e8ec377463a6c188753d35b8fcec300a658e.svg?branch=master&step=Julia%201.6-nightly
[buildkite-julianightly-img]: https://badge.buildkite.com/1509baa1122772e8ec377463a6c188753d35b8fcec300a658e.svg?branch=master&step=Julia%20nightly
[buildkite-url]: https://buildkite.com/julialang/kernelabstractions-dot-jl

License
-------

KernelAbstractions.jl is licensed under the [MIT license](LICENSE.md).
