module LinearAlgebraExt

using KernelAbstractions: KernelAbstractions
using LinearAlgebra: Tridiagonal, Diagonal

KernelAbstractions.get_backend(A::Diagonal) = KernelAbstractions.get_backend(A.diag)
KernelAbstractions.get_backend(A::Tridiagonal) = KernelAbstractions.get_backend(A.d)

end
