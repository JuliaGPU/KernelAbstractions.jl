module SparseArraysExt

using KernelAbstractions: KernelAbstractions
using SparseArrays: AbstractSparseArray, rowvals

function KernelAbstractions.get_backend(A::AbstractSparseArray)
    return KernelAbstractions.get_backend(rowvals(A))
end

end
