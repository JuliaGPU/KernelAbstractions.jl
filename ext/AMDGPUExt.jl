module AMDGPUExt

import KernelAbstractions
import AMDGPU

KernelAbstractions.fold_threshold(::AMDGPU.ROCBackend) = 16

end
