# KernelInterface

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliagpu.github.io/KernelAbstractions.jl/dev/kernelinterface/)

KernelInterface (or `KI`) defines the low-level API that backends implement to
provide device- and host-side functionality for
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

KernelInterface focuses on the lower-level functionality shared amongst
backends such as kernel launching, device intrinsics, and host-side
operations such as allocation and synchronization.


## License

KernelInterface is licensed under the [MIT license](LICENSE.md).
