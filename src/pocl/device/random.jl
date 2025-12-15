## random number generation

using Random
import RandomNumbers

# local memory with the actual seed, per subgroup, set by `initialize_rng_state`` or overridden by calling `seed!`
@inline function global_random_keys()
    n = get_num_sub_groups()
    ptr = random_keys()::LLVMPtr{UInt32, AS.Workgroup}
    return CLDeviceArray{UInt32, 1, AS.Workgroup}((n,), ptr)
end

# local memory with per-subgroup counters, incremented when generating numbers
@inline function global_random_counters()
    n = get_num_sub_groups()
    ptr = random_counters()::LLVMPtr{UInt32, AS.Workgroup}
    return CLDeviceArray{UInt32, 1, AS.Workgroup}((n,), ptr)
end

# initialization function, called automatically at the start of each kernel
function initialize_rng_state()
    subgroup_id = get_sub_group_id()
    @inbounds global_random_keys()[subgroup_id] = kernel_state().random_seed
    @inbounds global_random_counters()[subgroup_id] = 0
end

# generators

using Random123: philox2x_round, philox2x_bumpkey

# GPU-compatible/optimized version of the generator from Random123.jl
struct Philox2x32{R} <: RandomNumbers.AbstractRNG{UInt64} end

# default to 7 rounds; enough to pass SmallCrush
@inline Philox2x32() = Philox2x32{7}()

@inline function Base.getproperty(rng::Philox2x32, field::Symbol)
    subgroup_id = get_sub_group_local_id()

    if field === :key
        @inbounds global_random_keys()[subgroup_id]
    elseif field === :ctr1
        @inbounds global_random_counters()[subgroup_id]
    elseif field === :ctr2
        unsafe_trunc(UInt32, get_global_linear_id())
    end
end

@inline function Base.setproperty!(rng::Philox2x32, field::Symbol, x)
    subgroup_id = get_sub_group_local_id()

    if field === :key
        @inbounds global_random_keys()[subgroup_id] = x
    elseif field === :ctr1
        @inbounds global_random_counters()[subgroup_id] = x
    end
    return rng
end

@device_override @inline Random.default_rng() = Philox2x32()

"""
    Random.seed!(rng::Philox2x32, seed::Integer, [counter::Integer=0])

Seed the on-device Philox2x32 generator with an UInt32 number.
Should be called by at least one thread per warp.
"""
function Random.seed!(rng::Philox2x32, seed::Integer, counter::Integer=UInt32(0))
    rng.key = seed % UInt32
    rng.ctr1 = counter
    return
end

# seeding the implicit default RNG
@static if VERSION >= v"1.11-"
    @device_override Random.seed!(seed) =
        Random.seed!(Random.default_rng(), seed)
else
    @device_override Random.seed!(::Random._GLOBAL_RNG, seed) =
        Random.seed!(Random.default_rng(), seed)
end

@static if VERSION >= v"1.11-"
    # `Random.seed!(::AbstractRNG)` now passes a `nothing` seed value
    # TODO: I don't think there is any way in OpenCL to make this nondeterministic
    Random.seed!(rng::Philox2x32, seed::Nothing) =
        Random.seed!(rng, kernel_state().random_seed)
else
    # ... where it used to call `Random_make_seed()`
    @device_override Random.make_seed() = kernel_state().random_seed
end

"""
    Random.rand(rng::Philox2x32, UInt32)

Generate a byte of random data using the on-device Tausworthe generator.
"""
function Random.rand(rng::Philox2x32{R},::Type{UInt64}) where {R}
    ctr1, ctr2, key = rng.ctr1, rng.ctr2, rng.key

    if R > 0                               ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 1  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 2  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 3  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 4  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 5  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 6  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 7  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 8  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 9  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 10 key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 11 key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 12 key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 13 key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 14 key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 15 key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end

    # update the warp counter
    # NOTE: this performs the same update on every thread in the warp, but each warp writes
    #       to a unique location so the duplicate writes are innocuous
    # XXX: what if this overflows? we can't increment ctr2. bump the key?
    rng.ctr1 += Int32(1)

    # NOTE: it's too expensive to keep both numbers around in case the user only wanted one,
    #       so just make our 2x32 generator return 64-bit numbers by default.
    return (ctr1 % UInt64) << 32 | (ctr2 % UInt64)
end



# a hacky method of exposing constant tables as constant GPU memory

function emit_constant_array(name::Symbol, data::AbstractArray{T}) where {T}
    @dispose ctx=Context() begin
        T_val = convert(LLVMType, T)
        T_ptr = convert(LLVMType, LLVMPtr{T,AS.UniformConstant})

        # define function and get LLVM module
        llvm_f, _ = create_function(T_ptr)
        mod = LLVM.parent(llvm_f)

        # create a global memory global variable
        # TODO: global_var alignment?
        T_global = LLVM.ArrayType(T_val, length(data))
        # XXX: why can't we use a single name like emit_shmem
        gv = GlobalVariable(mod, T_global, "gpu_$(name)_data", AS.UniformConstant)
        linkage!(gv, LLVM.API.LLVMInternalLinkage)
        initializer!(gv, ConstantArray(data))
        alignment!(gv, 16)

        # generate IR
        @dispose builder=IRBuilder() begin
            entry = BasicBlock(llvm_f, "entry")
            position!(builder, entry)

            ptr = gep!(builder, T_global, gv, [ConstantInt(0), ConstantInt(0)])

            untyped_ptr = bitcast!(builder, ptr, T_ptr)

            ret!(builder, untyped_ptr)
        end

        call_function(llvm_f, LLVMPtr{T,AS.UniformConstant})
    end
end

for var in [:ki, :wi, :fi, :ke, :we, :fe]
    val = getfield(Random, var)
    gpu_var = Symbol("gpu_$var")
    arr_typ = :(CLDeviceArray{$(eltype(val)),$(ndims(val)),AS.UniformConstant})
    @eval @inline @generated function $gpu_var()
        ptr = emit_constant_array($(QuoteNode(var)), $val)
        Expr(:call, $arr_typ, $(size(val)), ptr)
    end
end

## randn

@device_override @inline function Random.randn(rng::AbstractRNG)
    @label retry
    r = Random.rand(rng, Random.UInt52Raw())
    @inbounds begin
        r &= 0x000fffffffffffff
        rabs = Int64(r >> 1) # One bit for the sign
        idx = rabs & 0xFF
        x = ifelse(r % Bool, -rabs, rabs)*gpu_wi()[idx+1]
        rabs < gpu_ki()[idx+1] && return x # 99.3% of the time we return here 1st try
        # TODO: This code could be outlined once LLVM supports LDS access in recursively-called functions
        @inbounds if idx == 0
            while true
                xx = -Random.ziggurat_nor_inv_r*log(Random.rand(rng))
                yy = -log(Random.rand(rng))
                yy+yy > xx*xx &&
                    return (rabs >> 8) % Bool ? -Random.ziggurat_nor_r-xx : Random.ziggurat_nor_r+xx
            end
        elseif (gpu_fi()[idx] - gpu_fi()[idx+1])*Random.rand(rng) + gpu_fi()[idx+1] < exp(-0.5*x*x)
            return x # return from the triangular area
        else
            @goto retry
        end
    end
end

@device_override @inline function Random.randn(rng::AbstractRNG, ::Type{T}) where {T <: Union{Float16, Float32}}
    @invoke Random.randn(rng::AbstractRNG, T::Type{<:AbstractFloat})
end

## randexp

@device_override @inline function Random.randexp(rng::AbstractRNG)
    @label retry
    ri = Random.rand(rng, Random.UInt52Raw())
    @inbounds begin
        ri &= 0x000fffffffffffff
        idx = ri & 0xFF
        x = ri*gpu_we()[idx+1]
        ri < gpu_ke()[idx+1] && return x # 98.9% of the time we return here 1st try
        # TODO: This code could be outlined once LLVM supports LDS access in recursively-called functions
        @inbounds if idx == 0
            return Random.ziggurat_exp_r - log(Random.rand(rng))
        elseif (gpu_fe()[idx] - gpu_fe()[idx+1])*Random.rand(rng) + gpu_fe()[idx+1] < exp(-x)
            return x # return from the triangular area
        else
            @goto retry
        end
    end
end

@device_override @inline function Random.randexp(rng::AbstractRNG, ::Type{T}) where {T <: Union{Float16, Float32}}
    @invoke Random.randexp(rng::AbstractRNG, T::Type{<:AbstractFloat})
end

@device_override Random.Sampler(::Type{<:AbstractRNG}, r::AbstractUnitRange{T},
                                ::Random.Repetition) where {T<:Union{Int64, UInt64}} = Random.SamplerRangeFast(r)
