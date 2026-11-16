module oneAPIExt

import KernelAbstractions
import oneAPI

KernelAbstractions.fold_threshold(::oneAPI.oneAPIBackend) = 16

end
