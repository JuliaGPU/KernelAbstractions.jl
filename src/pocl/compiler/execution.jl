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
# completion, keeping arguments alive. queue waits yield; host launches yield only after
# team or long launches (see `host_dispatch`).
#
# work-groups extending beyond `extent` run `partial`, a bounds-checked variant with the
# same argument layout. both variants share arguments and exception status. queue launches
# use `partial` for all work-groups: separate queued kernels cannot share group ids.
# `groups` is the work-group count, if known.
function launch_and_wait(
        kernel::AbstractKernel, args::Tuple, partial = nothing, extent = nothing,
        groups = nothing; kwargs...
    )
    GC.@preserve args begin
        # prepare both variants before running either; if the unchecked variant is
        # unavailable, run every work-group checked.
        hp = host_kernel(with_fun(kernel, partial), values(kwargs), groups)
        h = hp === nothing ? nothing : full_host_kernel(kernel, partial, hp)
        if hp === nothing
            info = exception_info()
            info[] = ExceptionInfo_st()
            GC.@preserve info begin
                info_ptr = Base.unsafe_convert(Ptr{ExceptionInfo_st}, info)
                event = launch_tuple(with_fun(kernel, partial), args, info_ptr; kwargs...)
                wait_for_kernel(event)
            end
            check_exception(info)
        else
            global_size = values(kwargs).global_size
            num_groups = something(groups, global_size .÷ hp.local_size)
            extent = something(extent, global_size)
            # host launches finish before returning, even when interrupted
            info = Ref(ExceptionInfo_st())
            GC.@preserve info begin
                info_ptr = Base.unsafe_convert(Ptr{ExceptionInfo_st}, info)
                host_launch_converted(kernel, args, info_ptr, hp, num_groups, h, extent)
            end
            # passing the status avoids allocating `info`
            check_exception(info[].status)
        end
    end
    return nothing
end

# kept out of `launch_and_wait`, which is compiled for every kernel
@noinline function wait_for_kernel(event::cl.Event)
    try
        wait(event)
    finally
        cl.clReleaseEvent(event)
    end
    return
end

check_exception(info::Base.RefValue{ExceptionInfo_st}) = check_exception(info[].status)
@inline function check_exception(status::Int32)
    status == 0 || throw_kernel_exception()
    return
end
@noinline throw_kernel_exception() = throw(KernelException(device()))

@inline launch_tuple(
    kernel::AbstractKernel, args::Tuple, exception_info::Ptr;
    global_size = (1,), local_size = nothing
) = launch_converted(kernel, args, exception_info, global_size, local_size)

# kept out of the launches, which are compiled for every kernel: inlining `rand` makes them
# several times as expensive to compile
@inline kernel_state(rng_state::Bool, exception_info::Ptr) =
    KernelState(rng_state ? rng_seed() : UInt32(0), UInt64(UInt(exception_info)))
@noinline rng_seed() = Base.rand(UInt32)

# For the generators below: the argument types of a kernel compiled for `F` and the
# argument types `TT`, as a tuple type, and the expressions that compute the arguments from
# `kernel`, its `nargs` arguments `args` and `exception_info`, with `null` for the values of
# non-isbits arguments. (a function of its own, working on vectors, so that it is compiled
# once rather than for every number of arguments.)
function kernel_call_arguments(@nospecialize(F), @nospecialize(TT), nargs::Int, null)
    params = Any[F, TT.parameters...]   # Base.signature_type with a function type
    exprs = Any[:(kernel.f); Any[:(clconvert(args[$i])) for i in 1:nargs]]
    call_t = Type[KernelState]
    call_args = Any[:(kernel_state(kernel.rng_state, exception_info))]
    for (dt, ex) in zip(params, exprs)
        # filter out ghost arguments that shouldn't be passed
        pass_arg(dt) || continue
        # replace non-isbits arguments (they should be unused, or compilation would have
        # failed)
        push!(call_t, isbitstype(dt) ? dt : Ptr{Any})
        push!(call_args, isbitstype(dt) ? ex : null)
    end
    return Base.to_tuple_type(call_t), call_args
end

@inline @generated function launch_converted(
        kernel::AbstractKernel{F, TT}, args::Tuple, exception_info, global_size, local_size
    ) where {F, TT}
    call_tt, call_args = kernel_call_arguments(F, TT, fieldcount(args), :C_NULL)

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

# queue-fallback exception storage is per task, since each task waits for its launches. it
# must outlive the stack frame, as an interrupted wait leaves the kernel running.
@noinline exception_info() = get!(task_local_storage(), :POCLExceptionInfo) do
    Ref(ExceptionInfo_st())
end::Base.RefValue{ExceptionInfo_st}


## host-side kernels

struct HostKernel{F, TT} <: AbstractKernel{F, TT}
    f::F
    fun::cl.Kernel
    rng_state::Bool
end

# `kernel`, with `fun`, a variant of its compiled kernel for arguments of the same layout
@inline with_fun(kernel::AbstractKernel, ::Nothing) = kernel
@inline with_fun(kernel::HostKernel{F, TT}, fun::cl.Kernel) where {F, TT} =
    HostKernel{F, TT}(kernel.f, fun, kernel.rng_state)


## host-side API

const clfunction_lock = ReentrantLock()

# `compiled_tt` are the argument types to compile the kernel for, if they aren't `tt`. they
# may only differ in ways that don't change how the arguments are passed (e.g. in the type
# parameters of a struct). it isn't specialized on, so that looking up either kernel is the
# same code.
Base.@nospecializeinfer function clfunction(
        f::F, tt::TT = Tuple{}, @nospecialize(compiled_tt::Type) = tt; kwargs...
    ) where {F, TT}
    kw = values(kwargs)
    # besides the signature, the kernel only depends on the session and the compiler options.
    # the kernel compiled for other types has a key of another type, and thus a cache of its
    # own: launches alternating between both kernels then find either on the fast path.
    kernel, rng_state = if compiled_tt === tt
        GPUCompiler.cached_lookup(Tuple{cl.Kernel, Bool}, F, tt, (context(), kw)) do
            lookup_kernel(F, tt, kw)
        end
    else
        # (wrapped, so that the types of the key and the lookup don't depend on it)
        compiled = Some{Type}(compiled_tt)
        GPUCompiler.cached_lookup(Tuple{cl.Kernel, Bool}, F, tt, (context(), kw, compiled)) do
            lookup_kernel(F, something(compiled), kw)
        end
    end
    # not cached: that would keep every callable that was ever launched alive
    return HostKernel{F, tt}(f, kernel, rng_state)
end

# Returns the `cl.Kernel` for `F` and `tt` on the session's context, and whether it uses
# the device-side RNG. Nothing in here depends on the types of the kernel or its arguments,
# so it is inferred and compiled once, rather than for every new kernel.
@noinline Base.@nospecializeinfer function lookup_kernel(
        @nospecialize(F::Type), @nospecialize(tt::Type), @nospecialize(kwargs::NamedTuple)
    )
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

        return kernel::cl.Kernel, res.device_rng
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
# than specialized for every kernel.
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


## host kernels
#
# `cl_pocl_host_kernel` runs work-groups on the calling thread without queues or events.
# small launches stay on the launching task; larger ones use a persistent Julia task team
# (see `ThreadTeams.foreach_range`).

import ThreadTeams

# `foreach_range` limits for team launches and stealing, respectively. above these, run on
# the launching task or disable stealing (see `host_schedule`).
const host_team_max_groups = typemax(Int) >> 2
const host_steal_max_groups = Int(typemax(Int32))

# a team costs about 1 µs to start and join. a trivial kernel (store or axpy) needs about
# 8192 work-items to recover that cost on 4 or 8 threads (see `host_schedule`).
const host_kernel_serial_items = 8192

# host kernels use the calling thread's stack; Julia task stacks are at least 4 MiB. cap
# estimated use at 1 MiB, using PoCL's upper bound per work-item. tuning shrinks work-groups
# to fit (see `link_kernel`); sizes that still exceed the limit use the queue.
const host_kernel_stack_limit = 1024 * 1024

# the largest power-of-two work-group of `k` within `host_kernel_stack_limit` (1 if a single
# work-item exceeds it)
function host_kernel_max_items(k::cl.Kernel)
    k.private_mem_size == 0 && return typemax(Int)
    return prevpow(2, max(host_kernel_stack_limit ÷ k.private_mem_size, 1))
end

@inline function host_kernel(kernel::HostKernel, kwargs::NamedTuple, groups)
    haskey(kwargs, :local_size) && haskey(kwargs, :global_size) || return nothing
    local_size, global_size = kwargs.local_size, kwargs.global_size
    local_size isa Dims{3} && global_size isa Dims{3} || return nothing
    # host kernels only run whole work-groups
    groups === nothing && global_size .% local_size != (0, 0, 0) && return nothing
    return host_kernel(kernel.fun, local_size, kernel.rng_state)
end

# use the unchecked variant at `hp`'s work-group size, or keep `hp` if it is unavailable
@inline function full_host_kernel(kernel::HostKernel, partial, hp::cl.HostKernelHandle)
    (partial === nothing || partial === kernel.fun) && return hp
    return something(host_kernel(kernel.fun, hp.local_size, kernel.rng_state), hp)
end

# kept out of `launch_and_wait`, which is compiled for every kernel
@noinline function host_kernel(k::cl.Kernel, local_size::Dims{3}, rng_state::Bool)
    h = Base.@atomic :acquire k.host_kernel
    if h === nothing || h.local_size != local_size
        h = host_kernel_slow(k, local_size, rng_state)
    end
    return h.id == C_NULL ? nothing : h
end

@noinline function host_kernel_slow(k::cl.Kernel, local_size::Dims{3}, rng_state::Bool)
    h = @lock k.lock begin
        i = findfirst(h -> h.local_size == local_size, k.host_kernels)
        if i === nothing
            h = create_host_kernel(k, local_size, rng_state)
            push!(k.host_kernels, h)
            h
        else
            k.host_kernels[i]
        end
    end
    Base.@atomic :release k.host_kernel = h
    return h
end

# with `k.lock` held
function create_host_kernel(k::cl.Kernel, local_size::Dims{3}, rng_state::Bool)
    dev = device()
    stack = k.private_mem_size * prod(local_size)
    stack > host_kernel_stack_limit && return cl.HostKernelHandle(C_NULL, local_size)

    # the host kernel takes the sizes of local memory arguments from the kernel
    lsize = Csize_t[local_size...]
    if rng_state
        num_sub_groups = cl.KernelSubGroupInfo(k, dev, lsize).sub_group_count
        nargs = cl.num_args(k) - 2
        # one UInt32 per sub-group
        cl.set_arg!(k, nargs + 1, cl.LocalMem(UInt32, num_sub_groups))
        cl.set_arg!(k, nargs + 2, cl.LocalMem(UInt32, num_sub_groups))
    end

    err = Ref{Int32}()
    id = cl.clCreateHostKernelPOCL(k, dev, 3, lsize, err)
    # don't remember transient failures
    err[] == cl.CL_OUT_OF_HOST_MEMORY && throw(OutOfMemoryError())
    if err[] != cl.CL_SUCCESS
        # e.g. the kernel needs more local memory than a host kernel can have
        @debug "Kernel can't run on Julia's threads, using PoCL's" exception = cl.CLError(err[])
        return cl.HostKernelHandle(C_NULL, local_size)
    end
    scratch_size, scratch_alignment = Ref{Csize_t}(), Ref{Csize_t}()
    cl.clGetHostKernelInfoPOCL(
        id, cl.CL_HOST_KERNEL_SCRATCH_SIZE_POCL, sizeof(Csize_t), scratch_size, C_NULL
    )
    cl.clGetHostKernelInfoPOCL(
        id, cl.CL_HOST_KERNEL_SCRATCH_ALIGNMENT_POCL, sizeof(Csize_t), scratch_alignment, C_NULL
    )
    return cl.HostKernelHandle(id, local_size, Int(scratch_size[]), Int(scratch_alignment[]))
end

# per-thread local memory and printf buffer. runs must not yield: they share scratch storage
# and must not overlap. look up the thread after the last possible yield, which may migrate
# the task. only the owning thread mutates its `HostScratch`; adding threads replaces the
# slots vector.
mutable struct HostScratch
    ptr::Ptr{Cvoid}     # aligned
    raw::Ptr{Cvoid}
    size::Int
    events_run::UInt64  # see `run_other_tasks`
end

mutable struct HostScratchSlots
    Base.@atomic slots::Vector{HostScratch}
    const lock::ReentrantLock
end

const host_scratch = HostScratchSlots(HostScratch[], ReentrantLock())

@inline function host_scratch_ptr(size::Int, alignment::Int)
    s = host_scratch_slot()
    # (the alignment is a power of two)
    if s.size < size || UInt(s.ptr) & (alignment - 1) != 0
        grow_host_scratch!(s, size, alignment)
    end
    return s.ptr
end

@inline function host_scratch_slot()
    while true
        tid = Threads.threadid()
        slots = Base.@atomic :acquire host_scratch.slots
        tid <= length(slots) && return @inbounds slots[tid]
        # may yield, so look the thread up again afterwards
        add_host_scratch_slots()
    end
    return
end

# doesn't yield
@noinline function grow_host_scratch!(s::HostScratch, size::Int, alignment::Int)
    # alignment is a power of two, and at least that of malloc
    size = max(size, s.size, 64 * 1024)
    align = max(alignment, 64)
    raw = Libc.malloc(size + align)
    raw == C_NULL && throw(OutOfMemoryError())
    Libc.free(s.raw)
    s.raw = raw
    s.ptr = Ptr{Cvoid}((UInt(raw) + align - 1) & ~UInt(align - 1))
    s.size = size
    return
end

@noinline function add_host_scratch_slots()
    @lock host_scratch.lock begin
        slots = Base.@atomic :acquire host_scratch.slots
        n = Threads.maxthreadid()
        if n > length(slots)
            new = [slots; [HostScratch(C_NULL, C_NULL, 0, 0) for _ in (length(slots) + 1):n]]
            Base.@atomic :release host_scratch.slots = new
        end
    end
    return
end

# run `first_group:first_group + group_count - 1`; report PoCL traps through the launch's
# exception status
@inline function host_run(
        h::cl.HostKernelHandle, arg_values::Ptr{Ptr{Cvoid}}, num_groups::Ptr{Csize_t},
        first_group::Int, group_count::Int, exception_info::Ptr{ExceptionInfo_st},
        scratch::Ptr{Cvoid}
    )
    err = cl.clRunHostKernelPOCL(
        h.id, arg_values, num_groups, first_group, group_count, scratch
    )
    if err != cl.CL_SUCCESS
        status = reinterpret(Ptr{Int32}, exception_info)
        UnsafeAtomics.store!(status, Int32(err), UnsafeAtomics.release)
    end
    return
end

# a callable struct keeps team code shared across kernels. up to `threads` threads run
# ranges of work-groups, stealing if `steal` is set (see `run_team`).
#
# `full` runs work-groups entirely within the ndrange; `h` runs partial ones. both use the
# same argument layout. `full === h` when no unchecked variant is needed or available.
struct HostLaunch
    h::cl.HostKernelHandle
    full::cl.HostKernelHandle
    arg_values::Ptr{Ptr{Cvoid}}
    num_groups::Ptr{Csize_t}
    groups::Dims{3}
    partial::NTuple{3, Bool}
    total::Int
    threads::Int
    steal::Bool
    scratch_size::Int
    scratch_alignment::Int
    exception_info::Ptr{ExceptionInfo_st}
end

# batch `first:last-1` into runs of full or partial work-groups. x varies fastest: a
# partial x dimension affects each row's last group; partial y or z dimensions affect whole
# rows.
function host_run(l::HostLaunch, first::Int, last::Int, scratch::Ptr{Cvoid})
    if l.full === l.h
        host_run(l.h, l.arg_values, l.num_groups, first, last - first, l.exception_info, scratch)
        return
    end
    gx, gy, gz = l.groups
    px, py, pz = l.partial
    g = first
    while g < last
        row = g ÷ gx
        x = g - row * gx
        y, z = row % gy, row ÷ gy
        if pz && z == gz - 1
            # the rest of the launch
            h, stop = l.h, l.total
        elseif py && y == gy - 1
            h, stop = l.h, (row + 1) * gx
        elseif px && x == gx - 1
            h, stop = l.h, g + 1
        else
            # up to the next partial work-group
            h = l.full
            stop = px ? row * gx + gx - 1 :
                py ? (z * gy + gy - 1) * gx :
                pz ? (gz - 1) * gy * gx : l.total
        end
        stop = min(stop, last)
        host_run(h, l.arg_values, l.num_groups, g, stop - g, l.exception_info, scratch)
        g = stop
    end
    return
end

# `foreach_range` uses inclusive, 1-based ranges. time the range containing group 1 and
# scale to all groups to estimate the launch's serial duration.
function (l::HostLaunch)(first::Int, last::Int)
    scratch = host_scratch_ptr(l.scratch_size, l.scratch_alignment)
    if first != 1
        host_run(l, first - 1, last, scratch)
        return
    end
    start = cycle_counter()
    host_run(l, 0, last, scratch)
    duration = cycle_counter() - start
    # (in floating point: the product may not fit)
    sample = unsafe_trunc(UInt64, min(Float64(duration) * (l.total / last), 0x1p63))
    record_serial_sample(l.h, l.total, sample)
    return
end

# racy by design: these statistics affect scheduling policy only
@inline function record_serial_sample(h::cl.HostKernelHandle, groups::Int, sample::UInt64)
    if (Base.@atomic :monotonic h.serial_groups_a) == groups
        previous = Base.@atomic :monotonic h.serial_sample_a
    else
        previous = (Base.@atomic :monotonic h.serial_groups_b) == groups ?
            (Base.@atomic :monotonic h.serial_sample_b) : UInt64(0)
        # move the previous grid to slot b
        g = Base.@atomic :monotonic h.serial_groups_a
        sample_a = Base.@atomic :monotonic h.serial_sample_a
        duration_a = Base.@atomic :monotonic h.serial_duration_a
        Base.@atomic :monotonic h.serial_groups_b = g
        Base.@atomic :monotonic h.serial_sample_b = sample_a
        Base.@atomic :monotonic h.serial_duration_b = duration_a
        Base.@atomic :monotonic h.serial_groups_a = groups
    end
    Base.@atomic :monotonic h.serial_sample_a = sample
    Base.@atomic :monotonic h.serial_duration_a = min(sample, previous)
    return
end

# the lesser of the last two estimates for a grid of `groups` work-groups, zero if unknown
@inline function serial_duration(h::cl.HostKernelHandle, groups::Int)
    (Base.@atomic :monotonic h.serial_groups_a) == groups &&
        return Base.@atomic :monotonic h.serial_duration_a
    (Base.@atomic :monotonic h.serial_groups_b) == groups &&
        return Base.@atomic :monotonic h.serial_duration_b
    return UInt64(0)
end

# defer interrupts until all work-groups finish. otherwise a team skips unstarted ranges,
# and a serial launch can stop between full and partial groups.
function (l::HostLaunch)()
    if l.threads == 1
        uninterruptible(run_serial, l)
    else
        uninterruptible(run_team, l)
    end
    return
end

run_serial(l::HostLaunch) = l(1, l.total)

# assign blocks as in `Threads.@threads :static`, independent of the launching thread, to
# preserve cache and NUMA locality across equal-sized grids. stealing prevents slow or
# descheduled threads from holding up the launch. a busy team (including nested launches and
# `Threads.@threads` loops) runs the launch on the calling task.
run_team(l::HostLaunch) = ThreadTeams.foreach_range(
    l, l.total; threads = l.threads, schedule = l.steal ? :steal : :static
)

# `f(x)`, with interrupts and (on Julia 1.14+) cancellation delivered after it returns
@inline function uninterruptible(f::F, x) where {F}
    @static if isdefined(Base, :CANCEL_TOKEN)
        if Base.CANCEL_TOKEN[] === nothing
            f(x)
        else
            Base.ScopedValues.with(() -> f(x), Base.CANCEL_TOKEN => nothing)
            Base.checkcancel(Base.CANCEL_TOKEN[])
        end
    else
        # `disable_sigint`, without a closure
        Base.sigatomic_begin()
        try
            f(x)
        finally
            Base.sigatomic_end()
        end
    end
    return
end

# shared across kernels: avoid inlining and specialization. `h` runs partial work-groups,
# `full` the rest (see `launch_and_wait`).
@noinline function host_dispatch(
        h::cl.HostKernelHandle, arg_values::Ptr{Ptr{Cvoid}}, num_groups::Dims{3},
        exception_info::Ptr{ExceptionInfo_st}, full::cl.HostKernelHandle, extent::Dims{3}
    )
    total = prod(num_groups)
    total == 0 && return
    partial = extent .< num_groups .* h.local_size
    any(partial) || (full = h)
    scratch_size = max(h.scratch_size, full.scratch_size)
    scratch_alignment = max(h.scratch_alignment, full.scratch_alignment)
    groups_ref = Ref(Csize_t.(num_groups))
    GC.@preserve groups_ref begin
        groups_ptr = Ptr{Csize_t}(Base.unsafe_convert(Ptr{NTuple{3, Csize_t}}, groups_ref))
        launch = HostLaunch(
            h, full, arg_values, groups_ptr, num_groups, partial, total, 1, false,
            scratch_size, scratch_alignment, exception_info
        )
        # finalizers can't switch tasks, which waiting for other threads and taking the
        # lock of the scratch slots may do
        if GC.in_finalizer()
            host_dispatch_finalizer(launch)
            return
        end
        s = session()
        long = (Base.@atomic :monotonic h.last_duration) >= s.host_kernel_long_ticks
        threads, steal = host_schedule(h, s, total, long)
        if threads == 1 && !long
            if full === h
                run_short(launch)
            else
                uninterruptible(run_short, launch)
            end
            return
        end
        start = cycle_counter()
        launch = HostLaunch(
            h, full, arg_values, groups_ptr, num_groups, partial, total, threads, steal,
            scratch_size, scratch_alignment, exception_info
        )
        launch()
        Base.@atomic :monotonic h.last_duration = cycle_counter() - start
        Base.@atomic :monotonic h.last_groups = total
    end
    # yield after team or long launches so repeated launches don't starve tasks, timers or
    # I/O
    run_other_tasks()
    return
end

# service waiting tasks; otherwise poll timers and I/O at most once per millisecond (the
# timer resolution). polling and yielding each cost 12 µs on macOS. track polls per thread:
# only one thread services events, outside `Threads.@threads`.
@noinline function run_other_tasks()
    if !other_tasks()
        s = host_scratch_slot()
        now = time_ns()
        now - s.events_run < 1_000_000 && return
        s.events_run = now
        Base.process_events()
        other_tasks() || return
    end
    yield()
    return
end

@inline function other_tasks()
    @static if isdefined(Base, :workqueue_for) && isdefined(Base, :Partr) &&
            isdefined(Base.Partr, :multiq_check_empty)
        return !isempty(Base.workqueue_for(Threads.threadid())) ||
            !Base.Partr.multiq_check_empty()
    else
        return true
    end
end

# choose the thread count and stealing policy. only the `foreach_range` limits constrain
# correctness.
#
# small grids stay on one thread unless both recent serial estimates reach the threshold.
# estimates are per grid because work per item can vary with grid size. timing the first
# range also gives an estimate for team launches. switching between serial and team launches
# moves data between caches; keep the threshold well above trivial-kernel times to avoid
# repeated switching.
#
# use static blocks after a short launch of the same grid. first or long launches allow
# stealing so slow or descheduled threads don't hold up the team. both policies start blocks
# on the same threads, preserving locality.
function host_schedule(h::cl.HostKernelHandle, s::Session, total::Int, long::Bool)
    threads = min(total, s.limits.compute_units)
    (threads > 1 && total <= host_team_max_groups) || return 1, false
    # (dividing: the number of work-items may not fit)
    if total <= host_kernel_serial_items ÷ prod(h.local_size) &&
            serial_duration(h, total) < s.host_kernel_serial_ticks
        return 1, false
    end
    steal = long || (Base.@atomic :monotonic h.last_groups) != total
    return threads, steal && total <= host_steal_max_groups
end

# a short launch on the launching task: one run covers the whole launch, so one timing
# supplies both the serial estimate and the last-launch duration
@inline function run_short(l::HostLaunch)
    scratch = host_scratch_ptr(l.scratch_size, l.scratch_alignment)
    start = cycle_counter()
    host_run(l, 0, l.total, scratch)
    duration = cycle_counter() - start
    record_serial_sample(l.h, l.total, duration)
    Base.@atomic :monotonic l.h.last_duration = duration
    Base.@atomic :monotonic l.h.last_groups = l.total
    return
end

# private scratch avoids locks and task switches, which finalizers cannot perform
@noinline function host_dispatch_finalizer(l::HostLaunch)
    # alignment is a power of two, and at least that of malloc
    align = l.scratch_alignment
    raw = Libc.malloc(l.scratch_size + align)
    raw == C_NULL && throw(OutOfMemoryError())
    scratch = Ptr{Cvoid}((UInt(raw) + align - 1) & ~UInt(align - 1))
    try
        host_run(l, 0, l.total, scratch)
    finally
        Libc.free(raw)
    end
    return
end

# like `launch_converted`, but passes a pointer to each converted argument
@inline @generated function host_launch_converted(
        kernel::AbstractKernel{F, TT}, args::Tuple, exception_info, h, num_groups, full,
        extent
    ) where {F, TT}
    call_tt, call_args = kernel_call_arguments(F, TT, fieldcount(args), :(Ptr{Any}(C_NULL)))
    N = length(call_args)
    arg_ptrs = Expr[:(base + $(fieldoffset(call_tt, i))) for i in 1:N]

    # the converted arguments only hold pointers to the arrays in `args`. the local memory
    # arguments of RNG kernels are set on the host kernel, so they get no value.
    return quote
        vals = Ref{$call_tt}(($(call_args...),))
        GC.@preserve args vals begin
            base = Ptr{Cvoid}(Base.unsafe_convert(Ptr{$call_tt}, vals))
            ptrs = Ref{NTuple{$(N + 2), Ptr{Cvoid}}}(($(arg_ptrs...), C_NULL, C_NULL))
            GC.@preserve ptrs begin
                arg_values = Ptr{Ptr{Cvoid}}(Base.unsafe_convert(Ptr{NTuple{$(N + 2), Ptr{Cvoid}}}, ptrs))
                host_dispatch(h, arg_values, num_groups, exception_info, full, extent)
            end
        end
        nothing
    end
end
