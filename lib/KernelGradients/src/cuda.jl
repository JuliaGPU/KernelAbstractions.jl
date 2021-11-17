import .CUDAKernels: CUDACtx, CUDACTX
import Cassette
import Enzyme

@inline function Cassette.overdub(::CUDACtx, ::typeof(Enzyme.autodiff_deferred), f, annotation::Enzyme.Annotation, args...)
    f′ = (args...) -> (Base.@_inline_meta; Cassette.overdub(CUDACTX, f, args...))
    Enzyme.autodiff_deferred(f′, annotation, args...)
end

@inline function Cassette.overdub(::CUDACtx, ::typeof(Enzyme.autodiff_deferred), f, args...)
    f′ = (args...) -> (Base.@_inline_meta; Cassette.overdub(CUDACTX, f, args...))
    Enzyme.autodiff_deferred(f′, args...)
end