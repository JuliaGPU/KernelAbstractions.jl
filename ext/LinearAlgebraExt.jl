module LinearAlgebraExt

using KernelAbstractions: KernelAbstractions
if isdefined(Base, :get_extension)
    using LinearAlgebra: Tridiagonal, Diagonal
else
    using ..LinearAlgebra: Tridiagonal, Diagonal
end

KernelAbstractions.get_backend(A::Diagonal) = KernelAbstractions.get_backend(A.diag)
KernelAbstractions.get_backend(A::Tridiagonal) = KernelAbstractions.get_backend(A.d)

end
