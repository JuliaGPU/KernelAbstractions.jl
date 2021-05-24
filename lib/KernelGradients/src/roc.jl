import .ROCKernels: ROCCTX, ROCCtx
import Cassette
import Enzyme

@inline function Cassette.overdub(::ROCCtx, ::typeof(Enzyme.autodiff_deferred), f, annotation::Enzyme.Annotation, args...)
    f′ = (args...) -> (Base.@_inline_meta; Cassette.overdub(ROCCTX, f, args...))
    Enzyme.autodiff_deferred(f′, annotation, args...)
end

@inline function Cassette.overdub(::ROCCtx, ::typeof(Enzyme.autodiff_deferred), f, args...)
    f′ = (args...) -> (Base.@_inline_meta; Cassette.overdub(ROCCTX, f, args...))
    Enzyme.autodiff_deferred(f′, args...)
end