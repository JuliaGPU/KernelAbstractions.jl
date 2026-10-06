export @opencl, clfunction, clconvert


## high-level @opencl interface

const MACRO_KWARGS = [:launch]
const COMPILER_KWARGS = [:kernel, :name, :always_inline, :debug_level, :validate, :sub_group_size]
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
    @gensym f_var kernel_f kernel_tt kernel

    # convert the arguments, call the compiler and launch the kernel
    # while keeping the original arguments alive
    push!(
        code.args,
        quote
            $f_var = $f
            GC.@preserve $(vars...) $f_var begin
                $kernel_f = $clconvert($f_var)
                $kernel_tt = $argument_types(($(var_exprs...),))
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
# (on Julia 1.14, the function type parameter is `Core.TypeEgal{T} <: Type{T}`)
Adapt.adapt_structure(
    to::KernelAdaptor,
    bc::Broadcast.Broadcasted{Style, <:Any, <:Type{T}}
) where {Style, T} =
    Broadcast.Broadcasted{Style}((x...) -> T(x...), adapt(to, bc.args), bc.axes)

# functions that capture a type, e.g., `Base.Fix1(convert, T)` as used by LinearAlgebra,
# which isn't a valid kernel argument either
function Adapt.adapt_structure(to::KernelAdaptor, f::Base.Fix1{<:Any, <:Type{T}}) where {T}
    g = adapt(to, f.f)
    return (x...) -> g(T, x...)
end
function Adapt.adapt_structure(to::KernelAdaptor, f::Base.Fix2{<:Any, <:Type{T}}) where {T}
    g = adapt(to, f.f)
    return (x...) -> g(x..., T)
end

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

# `Tuple{map(x -> Core.Typeof(clconvert(x)), args)...}`, without `map`, which isn't type
# stable for 32 or more elements
@inline @generated function argument_types(args::Tuple)
    types = (:(Core.Typeof(clconvert(args[$i]))) for i in 1:fieldcount(args))
    return :(Tuple{$(types...)})
end


## abstract kernel functionality

abstract type AbstractKernel{F, TT} end

pass_arg(@nospecialize dt) = !(GPUCompiler.isghosttype(dt) || Core.Compiler.isconstType(dt))

# The arguments are passed on as a tuple: Julia doesn't turn a splat of more than 32
# elements into a direct call, and a method with both varargs and keyword arguments splats
# them into its body. So the keyword method is defined explicitly.
(kernel::AbstractKernel)(args::Vararg{Any, N}) where {N} = launch_and_wait(kernel, args)
Core.kwcall(kwargs::NamedTuple, kernel::AbstractKernel, args::Vararg{Any, N}) where {N} =
    launch_and_wait(kernel, args; kwargs...)

# kernels operate on plain `Array`s, whose uses can't synchronize, so every launch waits for
# its kernel. this also keeps the arguments alive while the kernel runs. waiting yields to
# other tasks, as `synchronize` should (see the documentation on its semantics).
function launch_and_wait(kernel::AbstractKernel, args::Tuple; kwargs...)
    info = exception_info()
    info[] = ExceptionInfo_st()
    GC.@preserve args info begin
        event = launch_tuple(kernel, args, Base.unsafe_convert(Ptr{ExceptionInfo_st}, info); kwargs...)
        try
            wait(event)
        finally
            cl.clReleaseEvent(event)
        end
    end
    info[].status == 0 || throw(KernelException(device()))
    return nothing
end

@inline launch_tuple(
    kernel::AbstractKernel, args::Tuple, exception_info::Ptr;
    global_size = (1,), local_size = nothing
) = launch_converted(kernel, args, exception_info, global_size, local_size)

@inline @generated function launch_converted(
        kernel::AbstractKernel{F, TT}, args::Tuple, exception_info, global_size, local_size
    ) where {F, TT}
    sig = Tuple{F, TT.parameters...}    # Base.signature_type with a function type
    args = (:(kernel.f), (:(clconvert(args[$i])) for i in 1:fieldcount(args))...)

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
    pushfirst!(
        call_args,
        :(KernelState(kernel.rng_state ? Base.rand(UInt32) : UInt32(0), UInt64(UInt(exception_info))))
    )

    # finalize types
    call_tt = Base.to_tuple_type(call_t)

    # the converted arguments only hold pointers to the arrays in `args`
    return quote
        GC.@preserve args begin
            $cl.clcall(kernel.fun, $call_tt, ($(call_args...),); global_size, local_size, kernel.rng_state)
        end
    end
end


## exceptions

"""
    KernelException

An exception thrown during kernel execution on device `dev`. The kernel prints details about
the exception when it occurs, depending on the debug level (see Julia's `-g` option).
"""
struct KernelException <: Exception
    dev::cl.Device
end

Base.showerror(io::IO, err::KernelException) =
    print(io, "KernelException: exception thrown during kernel execution on device ", err.dev.name)

# where kernels report exceptions: per task, as a task waits for every kernel it launches
exception_info() = get!(task_local_storage(), :POCLExceptionInfo) do
    Ref(ExceptionInfo_st())
end::Base.RefValue{ExceptionInfo_st}


## host-side kernels

struct HostKernel{F, TT} <: AbstractKernel{F, TT}
    f::F
    fun::cl.Kernel
    rng_state::Bool
end


## host-side API

const clfunction_lock = ReentrantLock()

function clfunction(f::F, tt::TT = Tuple{}; kwargs...) where {F, TT}
    Base.@lock clfunction_lock begin
        config = compiler_config(device(); kwargs...)::OpenCLCompilerConfig
        source = methodinstance(F, tt)
        job = CompilerJob(source, config)

        res = compile_or_lookup(job)::OpenCLResults

        # Resolve the cl.Kernel for the session's context. There's one context per
        # session, so this is one `===` compare.
        ctx = context()
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
                # kernels for other contexts are from before a reset of the session
                empty!(res.kernels)
                push!(res.kernels, (ctx, linked))
            end
            linked
        else
            cached
        end

        # not cached: that would keep every callable that was ever launched alive
        return HostKernel{F, tt}(f, kernel, res.device_rng)
    end
end

# Look up cached compile artifacts for `job`, compiling on miss. Storage is managed
# by `GPUCompiler.cached_results` (Julia's integrated code cache on 1.11+, which also
# persists artifacts through precompilation; a session-local store on 1.10).
#
# `cached_results` returns `nothing` until code exists for the job; `obj === nothing`
# then identifies an `OpenCLResults` that hasn't been compiled yet. Compiling populates
# Julia's code cache, so the post-compile `cached_results` re-fetch is guaranteed to
# succeed. Every lookup is reported to the `@device_code_*` hook, so reflection
# observes cached kernels without recompiling them.
# Keep this specialized so the caller can avoid boxing `CompilerJob`. Its type parameters
# only identify the target and compiler parameters, so this is bounded per back-end rather
# than specialized for every kernel; `@noinline` keeps the body out of each `clfunction`.
@noinline function compile_or_lookup(job::CompilerJob)::OpenCLResults
    GPUCompiler.run_compile_hook(job)
    res = GPUCompiler.cached_results(OpenCLResults, job)
    if res === nothing || res.obj === nothing
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
