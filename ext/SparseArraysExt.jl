module SparseArraysExt

using KernelAbstractions: KernelAbstractions
if isdefined(Base, :get_extension)
    using SparseArrays: AbstractSparseArray, rowvals
else
    using ..SparseArrays: AbstractSparseArray, rowvals
end

function KernelAbstractions.get_backend(A::AbstractSparseArray)
    return KernelAbstractions.get_backend(rowvals(A))
end

end
