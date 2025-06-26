module StaticArraysExt

import KernelAbstractions: get_backend, CPU
using StaticArrays: SizedArray, MArray, SArray

get_backend(A::SizedArray) = get_backend(A.data)
get_backend(::MArray) = CPU()
# TODO: It makes sense to pass SArray to the GPU backend, so we can't make a determination
# get_backend(::SArray) = CPU()

end
