module MetalExt

import KernelAbstractions
import Metal

KernelAbstractions.fold_threshold(::Metal.MetalBackend) = 16

end
