module Arrays

using ..KernelAbstractions

using Adapt
using Base.Broadcast
import Base.Broadcast: BroadcastStyle, Broadcasted, AbstractArrayStyle
using LinearAlgebra

export AbstractKernelArray

abstract type AbstractKernelArray{T, N} <: DenseArray{T, N} end

const AbstractKernelArraysStyle{N} = Broadcast.ArrayStyle{AbstractKernelArray{T,N}} where T

# Wrapper types otherwise forget that they are AbstractKernelArray
for (W, ctor) in Adapt.wrappers
    @eval begin
        BroadcastStyle(::Type{<:$W}) where {AT<:AbstractKernelArray{T,N} where {T,N}} = BroadcastStyle(AT)
        backend(::Type{<:$W}) where {AT<:AbstractKernelArray{T,N} where {T,N}} = backend(AT)
    end
end

# This Union is a hack. Ideally Base would have a
#     Transpose <: WrappedArray <: AbstractArray
# and we could define our methods in terms of
#     Union{AbstractKernelArray, WrappedArray{<:Any, <:AbstractKernelArray}}
@eval const DestAbstractKernelArray =
    Union{AbstractKernelArray,
          $((:($W where {AT <: AbstractKernelArray}) for (W, _) in Adapt.wrappers)...),
          Base.RefValue{<:AbstractKernelArray} }

# Ref is special: it's not a real wrapper, so not part of Adapt,
# but it is commonly used to bypass broadcasting of an argument
# so we need to preserve its dimensionless properties.
BroadcastStyle(::Type{Base.RefValue{AT}}) where {AT<:AbstractKernelArray} = typeof(BroadcastStyle(AT))(Val(0))
backend(::Type{Base.RefValue{AT}}) where {AT<:AbstractKernelArray} = backend(AT)
# but make sure we don't dispatch to the optimized copy method that directly indexes
function Broadcast.copy(bc::Broadcasted{<:Broadcast.ArrayStyle{AbstractKernelArray{T,0,}}}) where {T}
    ElType = Broadcast.combine_eltypes(bc.f, bc.args)
    isbitstype(ElType) || error("Cannot broadcast function returning non-isbits $ElType.")
    dest = copyto!(similar(bc, ElType), bc)
    # TODO this needs @allowscalar when disabling scalar indexing is enabled
    return dest[CartesianIndex()]  # 0D broadcast needs to unwrap results
end

@kernel function broadcast_kernel!(dest, bc′)
    let i = @index(Global, Linear)
        let I = CartesianIndex(CartesianIndices(dest)[i])
            @inbounds dest[I] = bc′[I]
        end
    end
end

# TODO need a better way of selecting the device
device(A) = typeof(A) <: Array ? CPU() : CUDA()

@inline function Base.copyto!(dest::DestAbstractKernelArray, bc::Broadcasted{Nothing})
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    bc′ = Broadcast.preprocess(dest, bc)

    threads = 256

    event = Event(device(dest))
    event = broadcast_kernel!(device(dest), threads)(dest, bc′, ndrange=length(dest), dependencies=event)
    wait(device(dest), event)

    return dest
end

# Base defines this method as a performance optimization, but we don't know how to do
# `fill!` in general for all `DestAbstractKernelArray` so we just go straight to the fallback
@inline Base.copyto!(dest::DestAbstractKernelArray, bc::Broadcasted{<:AbstractArrayStyle{0}}) =
    copyto!(dest, convert(Broadcasted{Nothing}, bc))

## map
allequal(x) = true
allequal(x, y, z...) = x == y && allequal(y, z...)

function Base.map!(f, y::DestAbstractKernelArray, xs::AbstractArray...)
    @assert allequal(size.((y, xs...))...)
    return y .= f.(xs...)
end

function Base.map(f, y::DestAbstractKernelArray, xs::AbstractArray...)
    @assert allequal(size.((y, xs...))...)
    return f.(y, xs...)
end

end # module
