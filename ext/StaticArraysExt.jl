module StaticArraysExt

import KernelAbstractions: get_backend, POCLBackend
using StaticArrays: SizedArray, MArray

get_backend(A::SizedArray) = get_backend(A.data)
get_backend(::MArray) = POCLBackend()

end
