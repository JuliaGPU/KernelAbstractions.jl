module KernelGradients

import Enzyme
import KernelAbstractions: Kernel

function Enzyme.autodiff(kernel::Kernel{<:Any, <:Any, <:Any, Fun}) where Fun
    f = kernel.f
    function df(ctx, args...)
        Enzyme.autodiff_deferred(f::Fun, Enzyme.Const, ctx, args...)
    end
    similar(kernel, df)
end

end # module
