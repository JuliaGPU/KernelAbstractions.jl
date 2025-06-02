module LinearAlgebraExt

using KernelAbstractions: KernelAbstractions
using LinearAlgebra: Tridiagonal, Diagonal

KernelAbstractions.get_backend(A::Diagonal) = KernelAbstractions.get_backend_recur(x -> x.diag, A)
KernelAbstractions.get_backend(A::Tridiagonal) = KernelAbstractions.get_backend_recur(x -> x.d, A)

end
