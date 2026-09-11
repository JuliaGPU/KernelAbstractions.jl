export @opencl, clfunction, clconvert


## high-level @opencl interface

const MACRO_KWARGS = [:launch]
const COMPILER_KWARGS = [:kernel, :name, :always_inline, :validate, :sub_group_size]
const LAUNCH_KWARGS = [:global_size, :local_size, :queue]

macro opencl(ex...)
    call = ex[end]
    kwargs = map(ex[1:(end - 1)]) do kwarg
        if kwarg isa Symbol
            :($kwarg = $kwarg)
        elseif Meta.isexpr(kwarg, :(=))
            kwarg
        else
            throw(ArgumentError("Invalid keyword argument '$kwarg'"))
        end
    end

    # destructure the kernel call
    Meta.isexpr(call, :call) || throw(ArgumentError("second argument to @opencl should be a function call"))
    f = call.args[1]
    args = call.args[2:end]

    code = quote end
    vars, var_exprs = assign_args!(code, args)

    # group keyword argument
    macro_kwargs, compiler_kwargs, call_kwargs, other_kwargs =
        split_kwargs(kwargs, MACRO_KWARGS, COMPILER_KWARGS, LAUNCH_KWARGS)
    if !isempty(other_kwargs)
        key, val = first(other_kwargs).args
        throw(ArgumentError("Unsupported keyword argument '$key'"))
    end

    # handle keyword arguments that influence the macro's behavior
    launch = true
    for kwarg in macro_kwargs
        key, val = kwarg.args
        if key == :launch
            isa(val, Bool) || throw(ArgumentError("`launch` keyword argument to @opencl should be a constant value"))
            launch = val::Bool
        else
            throw(ArgumentError("Unsupported keyword argument '$key'"))
        end
    end
    if !launch && !isempty(call_kwargs)
        error("@opencl with launch=false does not support launch-time keyword arguments; use them when calling the kernel")
    end

    # FIXME: macro hygiene wrt. escaping kwarg values (this broke with 1.5)
    #        we esc() the whole thing now, necessitating gensyms...
    @gensym f_var kernel_f kernel_args kernel_tt kernel

    # convert the arguments, call the compiler and launch the kernel
    # while keeping the original arguments alive
    push!(
        code.args,
        quote
            $f_var = $f
            GC.@preserve $(vars...) $f_var begin
                $kernel_f = $clconvert($f_var)
                $kernel_args = map($clconvert, ($(var_exprs...),))
                $kernel_tt = Tuple{map(Core.Typeof, $kernel_args)...}
                $kernel = $clfunction($kernel_f, $kernel_tt; $(compiler_kwargs...))
                if $launch
                    $kernel($(var_exprs...); $(call_kwargs...))
                end
                $kernel
            end
        end
    )

    return esc(
        quote
            let
                $code
            end
        end
    )
end


## argument conversion

struct KernelAdaptor
    svm_pointers::Union{Nothing, Vector{Ptr{Cvoid}}}
end

# # assume directly-passed pointers are SVM pointers
# function Adapt.adapt_storage(to::KernelAdaptor, ptr::Ptr{T}) where {T}
#     push!(to.svm_pointers, ptr)
#     return ptr
# end

# # convert SVM buffers to their GPU address
# function Adapt.adapt_storage(to::KernelAdaptor, buf::cl.SVMBuffer)
#     ptr = pointer(buf)
#     push!(to.svm_pointers, ptr)
#     return ptr
# end

# Base.RefValue isn't GPU compatible, so provide a compatible alternative
# TODO: port improvements from CUDA.jl
struct CLRefValue{T} <: Ref{T}
    x::T
end
Base.getindex(r::CLRefValue) = r.x
Adapt.adapt_structure(to::KernelAdaptor, r::Base.RefValue) = CLRefValue(adapt(to, r[]))

# broadcast sometimes passes a ref(type), resulting in a GPU-incompatible DataType box.
# avoid that by using a special kind of ref that knows about the boxed type.
struct CLRefType{T} <: Ref{DataType} end
Base.getindex(r::CLRefType{T}) where {T} = T
Adapt.adapt_structure(to::KernelAdaptor, r::Base.RefValue{<:Union{DataType, Type}}) =
    CLRefType{r[]}()

# case where type is the function being broadcasted
Adapt.adapt_structure(
    to::KernelAdaptor,
    bc::Broadcast.Broadcasted{Style, <:Any, Type{T}}
) where {Style, T} =
    Broadcast.Broadcasted{Style}((x...) -> T(x...), adapt(to, bc.args), bc.axes)

"""
    clconvert(x, [pointers])

This function is called for every argument to be passed to a kernel, allowing it to be
converted to a GPU-friendly format. By default, the function does nothing and returns the
input object `x` as-is.

Do not add methods to this function, but instead extend the underlying Adapt.jl package and
register methods for the the `OpenCL.KernelAdaptor` type.

The `pointers` argument is used to collect pointers to indirect SVM buffers, which need to
be registered with OpenCL before invoking the kernel.
"""
function clconvert(arg, pointers::Union{Nothing, Vector{Ptr{Cvoid}}} = nothing)
    return adapt(KernelAdaptor(pointers), arg)
end


## abstract kernel functionality

abstract type AbstractKernel{F, TT} end

pass_arg(@nospecialize dt) = !(GPUCompiler.isghosttype(dt) || Core.Compiler.isconstType(dt))

@inline @generated function (kernel::AbstractKernel{F, TT})(
        args::Vararg{Any, N};
        global_size = (1,), local_size = nothing
    ) where {F, TT, N}
    sig = Tuple{F, TT.parameters...}    # Base.signature_type with a function type
    args = (:(kernel.f), (:(clconvert(args[$i])) for i in 1:length(args))...)

    # filter out ghost arguments that shouldn't be passed
    to_pass = map(pass_arg, sig.parameters)
    call_t = Type[x[1] for x in zip(sig.parameters, to_pass) if x[2]]
    call_args = Union{Expr, Symbol}[x[1] for x in zip(args, to_pass)            if x[2]]

    # replace non-isbits arguments (they should be unused, or compilation would have failed)
    for (i, dt) in enumerate(call_t)
        if !isbitstype(dt)
            call_t[i] = Ptr{Any}
            call_args[i] = :C_NULL
        end
    end

    pushfirst!(call_t, KernelState)
    pushfirst!(call_args, :(KernelState(kernel.rng_state ? Base.rand(UInt32) : UInt32(0))))

    # finalize types
    call_tt = Base.to_tuple_type(call_t)

    return quote
        $cl.clcall(kernel.fun, $call_tt, $(call_args...); global_size, local_size, kernel.rng_state)
    end
end


## host-side kernels

struct HostKernel{F, TT} <: AbstractKernel{F, TT}
    f::F
    fun::cl.Kernel
    rng_state::Bool
end


## host-side API

const clfunction_lock = ReentrantLock()

# `HostKernel` with the world age and context it was resolved in; valid as long as no method
# has been defined since and the context is unchanged.
struct ResolvedKernel
    world::UInt
    context::nanoOpenCL.Context
    kernel::Any
end

# `HostKernel{F, tt}` instances keyed by their type
const _kernel_fastpath = Dict{DataType, ResolvedKernel}()

# Reading a `ScopedValue` allocates; outside of any dynamic scope it holds its default.
@static if VERSION >= v"1.11"
    @inline compile_hook_set() = Core.current_scope() !== nothing && GPUCompiler.compile_hook[] !== nothing
else
    @inline compile_hook_set() = GPUCompiler.compile_hook[] !== nothing
end

function clfunction(f::F, tt::TT = Tuple{}; kwargs...) where {F, TT}
    Base.@lock clfunction_lock begin
        ctx = context()
        world = Base.get_world_counter()
        cacheable = isempty(kwargs) && !compile_hook_set()
        if cacheable
            entry = get(_kernel_fastpath, HostKernel{F, tt}, nothing)
            if entry !== nothing && entry.world == world && entry.context === ctx
                return entry.kernel::HostKernel{F, tt}
            end
        end

        config = compiler_config(device(); kwargs...)::OpenCLCompilerConfig
        source = methodinstance(F, tt)
        job = CompilerJob(source, config)

        res = compile_or_lookup(job)::OpenCLResults

        # Resolve the cl.Kernel for the active context. Linear scan over the
        # session-local cache; almost always n=1, so this is one `===` compare.
        cached = nothing
        @inbounds for (cached_ctx, cached_kernel) in res.kernels
            if cached_ctx === ctx
                cached = cached_kernel
                break
            end
        end
        kernel = if cached === nothing
            linked = link_kernel(job, res.obj::Vector{UInt8}, res.entry::String)
            # Don't cache session-local kernel handles while precompiling: the
            # results struct is serialized into the package image along with its
            # CodeInstance, and the handles would come back dangling.
            if ccall(:jl_generating_output, Cint, ()) != 1
                push!(res.kernels, (ctx, linked))
            end
            linked
        else
            cached
        end

        h = hash(kernel, hash(f, hash(tt)))
        hostkernel = get!(_kernel_instances, h) do
            HostKernel{F, tt}(f, kernel, res.device_rng)
        end::HostKernel{F, tt}
        if cacheable
            _kernel_fastpath[HostKernel{F, tt}] = ResolvedKernel(world, ctx, hostkernel)
        end
        return hostkernel
    end
end

# Look up cached compile artifacts for `job`, compiling on miss. Storage is managed
# by `GPUCompiler.cached_results` (Julia's integrated code cache on 1.11+, which also
# persists artifacts through precompilation; a session-local store on 1.10).
#
# `cached_results` returns `nothing` until code exists for the job; `obj === nothing`
# then identifies an `OpenCLResults` that hasn't been compiled yet. Compiling populates
# Julia's code cache, so the post-compile `cached_results` re-fetch is guaranteed to
# succeed. The `compile_hook` check additionally forces the compile path so
# reflection-style consumers (`@device_code_*`) observe the compilation even on a hit.
function compile_or_lookup(@nospecialize(job::CompilerJob))::OpenCLResults
    res = GPUCompiler.cached_results(OpenCLResults, job)
    if res === nothing || res.obj === nothing || GPUCompiler.compile_hook[] !== nothing
        compiled = compile_to_obj(job)
        if res === nothing
            res = GPUCompiler.cached_results(OpenCLResults, job)::OpenCLResults
        end
        res.obj = compiled.obj
        res.entry = compiled.entry
        res.device_rng = compiled.device_rng
    end
    return res
end

# cache of kernel instances
const _kernel_instances = Dict{UInt, Any}()
