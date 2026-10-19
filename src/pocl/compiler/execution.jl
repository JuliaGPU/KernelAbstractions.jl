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
    h = host_kernel(kernel, values(kwargs))
    h === nothing || return host_launch_and_wait(kernel, h, args; kwargs...)
    return queue_launch_and_wait(kernel, args; kwargs...)
end

function queue_launch_and_wait(kernel::AbstractKernel, args::Tuple; kwargs...)
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

# The kernel `clfunction` last returned for a callable type, argument types and compiler
# keyword argument types. Looking it up again takes a lock and queries GPUCompiler's cache,
# which is most of the cost of launching a small kernel, so `clfunction` reuses it while the
# task's world age, the session's context and the keyword values are the same. It only holds
# the compiled artifacts, not the callable, which can hold large objects.
mutable struct KernelCacheSlot
    Base.@atomic entry::Any  # nothing, or (; world, context, kwargs, kernel, rng_state)
end

tls_world_age() = ccall(:jl_get_tls_world_age, UInt, ())

@generated function kernel_cache_slot(::Type{F}, ::Type{TT}, ::Type{K}) where {F, TT, K}
    slot = KernelCacheSlot(nothing)
    return :($slot)
end

@inline function lookup_cached_kernel(::Type{F}, tt::Type{TT}, kwargs::K) where {F, TT, K}
    # lookups report to `@device_code_*`, so don't bypass them while it is active
    GPUCompiler.active_compile_hook[] === nothing || return nothing
    entry = Base.@atomic :acquire kernel_cache_slot(F, TT, K).entry
    entry === nothing && return nothing
    entry = entry::@NamedTuple{
        world::UInt, context::cl.Context, kwargs::K, kernel::cl.Kernel, rng_state::Bool,
    }
    if entry.world == tls_world_age() && entry.context === context() &&
            isequal(entry.kwargs, kwargs)
        return entry
    end
    return nothing
end

function clfunction(f::F, tt::TT = Tuple{}; kwargs...) where {F, TT}
    cached = lookup_cached_kernel(F, tt, values(kwargs))
    cached === nothing || return HostKernel{F, tt}(f, cached.kernel, cached.rng_state)
    return clfunction_slow(f, tt; kwargs...)
end

@noinline function clfunction_slow(f::F, tt::TT; kwargs...) where {F, TT}
    world = tls_world_age()
    kernel = clfunction_uncached(f, tt; kwargs...)
    if ccall(:jl_generating_output, Cint, ()) != 1
        entry = (;
            world, context = context(), kwargs = values(kwargs), kernel = kernel.fun,
            kernel.rng_state,
        )
        Base.@atomic :release kernel_cache_slot(F, tt, typeof(values(kwargs))).entry = entry
    end
    return kernel
end

function clfunction_uncached(f::F, tt::TT = Tuple{}; kwargs...) where {F, TT}
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


## host kernels
#
# PoCL builds with the `cl_pocl_host_kernel` extension run a kernel's work-groups on the
# calling thread, without command queues, events or PoCL's thread pool. A launch then runs
# its work-groups on the launching task if there is one, and otherwise splits them into one
# range per thread, run by a team of Julia tasks that stay warm between launches.

import ThreadTeams

const host_team = ThreadTeams.TeamScheduler()
# long launches use more ranges than threads, which the threads take as they become free,
# so that a slower thread (e.g. an efficiency core) doesn't hold up the launch
const host_greedy_team = ThreadTeams.GreedyTeamScheduler()
const host_greedy_ranges_per_thread = 4

# work-groups run on the stack of the thread that runs them, Julia tasks' stacks are 4 MiB
# or more, and PoCL's estimate of a work-item's stack use is an upper bound. kernels that
# may need more run through the command queue on PoCL's threads instead.
const host_kernel_stack_limit = 1024 * 1024

# below this many work-items, a run isn't worth the transitions to and from GC-safe code
const host_kernel_gc_safe_items = 1 << 16

@inline host_kernel(kernel::AbstractKernel, kwargs) = nothing
@inline function host_kernel(kernel::HostKernel, kwargs::NamedTuple)
    haskey(kwargs, :local_size) && haskey(kwargs, :global_size) || return nothing
    local_size = kwargs.local_size
    local_size isa Dims{3} && kwargs.global_size isa Dims{3} || return nothing
    session().host_kernel_threads > 0 || return nothing
    k = kernel.fun
    h = Base.@atomic :acquire k.host_kernel
    if h === nothing || h.local_size != local_size
        h = host_kernel_slow(k, local_size, kernel.rng_state)
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
    stack = cl.work_group_info(k, dev).private_mem_size * prod(local_size)
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

# the scratch area (local memory and printf buffer) of each thread. a run doesn't yield, so
# runs on one thread never overlap, as long as the thread is looked up after the last point
# where the launching task may yield (and move to another thread). only the owning thread
# changes its `HostScratch`; growing the vector (for threads added at run time) replaces it.
mutable struct HostScratch
    ptr::Ptr{Cvoid}     # aligned
    raw::Ptr{Cvoid}
    size::Int
end

mutable struct HostScratchSlots
    Base.@atomic slots::Vector{HostScratch}
    const lock::ReentrantLock
end

const host_scratch = HostScratchSlots(HostScratch[], ReentrantLock())

@inline function host_scratch_ptr(h::cl.HostKernelHandle)
    while true
        tid = Threads.threadid()
        slots = Base.@atomic :acquire host_scratch.slots
        if tid <= length(slots)
            s = @inbounds slots[tid]
            if s.size < h.scratch_size || UInt(s.ptr) % h.scratch_alignment != 0
                grow_host_scratch!(s, h)
            end
            return s.ptr
        end
        # may yield, so look the thread up again afterwards
        add_host_scratch_slots()
    end
end

# doesn't yield
@noinline function grow_host_scratch!(s::HostScratch, h::cl.HostKernelHandle)
    # alignment is a power of two, and at least that of malloc
    size = max(h.scratch_size, s.size, 64 * 1024)
    align = max(h.scratch_alignment, 64)
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
            new = [slots; [HostScratch(C_NULL, C_NULL, 0) for _ in (length(slots) + 1):n]]
            Base.@atomic :release host_scratch.slots = new
        end
    end
    return
end

function host_launch_and_wait(
        kernel::HostKernel, h::cl.HostKernelHandle, args::Tuple; global_size, local_size
    )
    num_groups = map(div, global_size, local_size)
    if num_groups .* local_size != global_size
        # host kernels only run whole work-groups
        return queue_launch_and_wait(kernel, args; global_size, local_size)
    end
    info = exception_info()
    info[] = ExceptionInfo_st()
    GC.@preserve args info begin
        host_launch_converted(
            kernel, args, Base.unsafe_convert(Ptr{ExceptionInfo_st}, info), h, num_groups
        )
    end
    info[].status == 0 || throw(KernelException(device()))
    return nothing
end

# the work-groups `first_group:first_group + group_count - 1` of a launch. a failing run (a
# trap in PoCL) is reported through the launch's exception status.
@inline function host_run(
        h::cl.HostKernelHandle, arg_values::Ptr{Ptr{Cvoid}}, num_groups::Ptr{Csize_t},
        first_group::Int, group_count::Int, items::Int, exception_info::Ptr{ExceptionInfo_st},
        long::Bool
    )
    scratch = host_scratch_ptr(h)
    gc_safe = long || group_count * items >= host_kernel_gc_safe_items
    err = cl.clRunHostKernelPOCL(
        h.id, arg_values, num_groups, first_group, group_count, scratch, gc_safe
    )
    if err != cl.CL_SUCCESS
        status = reinterpret(Ptr{Int32}, exception_info)
        UnsafeAtomics.store!(status, Int32(err), UnsafeAtomics.release)
    end
    return
end

# a launch, run as one range of work-groups per thread. a callable struct rather than a
# closure: the team stores it in a slot per type without boxing.
struct HostLaunch
    h::cl.HostKernelHandle
    arg_values::Ptr{Ptr{Cvoid}}
    num_groups::Ptr{Csize_t}
    total::Int
    nchunks::Int
    items::Int
    exception_info::Ptr{ExceptionInfo_st}
    long::Bool
end

function (l::HostLaunch)(i::Int)
    first = ((i - 1) * l.total) ÷ l.nchunks
    last = (i * l.total) ÷ l.nchunks
    host_run(
        l.h, l.arg_values, l.num_groups, first, last - first, l.items, l.exception_info,
        l.long
    )
    return
end

function (l::HostLaunch)()
    if l.nchunks == 1
        l(1)
    elseif l.long
        ThreadTeams.tforeach(l, 1:l.nchunks; scheduler = host_greedy_team)
    else
        ThreadTeams.tforeach(l, 1:l.nchunks; scheduler = host_team)
    end
    return
end

@inline function host_dispatch(
        h::cl.HostKernelHandle, arg_values::NTuple{N, Ptr{Cvoid}}, num_groups::Dims{3},
        exception_info::Ptr{ExceptionInfo_st}
    ) where {N}
    total = prod(num_groups)
    total == 0 && return
    s = session()
    items = prod(h.local_size)
    nchunks = min(total, s.host_kernel_threads)
    # host kernels run on the launching task, which can't yield while they do. a launch
    # through the queue lets other tasks run while it waits, so do the same when the
    # previous launch was long: wait for a task that runs it.
    long = (Base.@atomic :monotonic h.last_duration) >= s.host_kernel_long_ticks
    # the greedy team runs on all of the pool's threads
    if long && nchunks > 1 && s.host_kernel_threads == ThreadTeams.nworkers()
        nchunks = min(total, host_greedy_ranges_per_thread * nchunks)
    end
    start = cycle_counter()
    args_ref = Ref(arg_values)
    groups_ref = Ref(Csize_t.(num_groups))
    GC.@preserve args_ref groups_ref begin
        args_ptr = Ptr{Ptr{Cvoid}}(Base.unsafe_convert(Ptr{NTuple{N, Ptr{Cvoid}}}, args_ref))
        groups_ptr = Ptr{Csize_t}(Base.unsafe_convert(Ptr{NTuple{3, Csize_t}}, groups_ref))
        launch = HostLaunch(
            h, args_ptr, groups_ptr, total, nchunks, items, exception_info, long
        )
        if long
            wait(Threads.@spawn launch())
        else
            launch()
        end
    end
    Base.@atomic :monotonic h.last_duration = cycle_counter() - start
    return
end

# like `launch_converted`, but passes a pointer to each converted argument
@inline @generated function host_launch_converted(
        kernel::AbstractKernel{F, TT}, args::Tuple, exception_info, h, num_groups
    ) where {F, TT}
    sig = Tuple{F, TT.parameters...}
    args = (:(kernel.f), (:(clconvert(args[$i])) for i in 1:fieldcount(args))...)

    to_pass = map(pass_arg, sig.parameters)
    call_t = Type[x[1] for x in zip(sig.parameters, to_pass) if x[2]]
    call_args = Union{Expr, Symbol}[x[1] for x in zip(args, to_pass) if x[2]]
    for (i, dt) in enumerate(call_t)
        if !isbitstype(dt)
            call_t[i] = Ptr{Any}
            call_args[i] = :(Ptr{Any}(C_NULL))
        end
    end
    pushfirst!(call_t, KernelState)
    pushfirst!(
        call_args,
        :(KernelState(kernel.rng_state ? Base.rand(UInt32) : UInt32(0), UInt64(UInt(exception_info))))
    )
    call_tt = Base.to_tuple_type(call_t)
    arg_ptrs = [:(base + $(fieldoffset(call_tt, i))) for i in 1:length(call_t)]

    # the converted arguments only hold pointers to the arrays in `args`. the local memory
    # arguments of RNG kernels are set on the host kernel, so they get no value.
    return quote
        vals = Ref{$call_tt}(($(call_args...),))
        GC.@preserve args vals begin
            base = Ptr{Cvoid}(Base.unsafe_convert(Ptr{$call_tt}, vals))
            if kernel.rng_state
                host_dispatch(h, ($(arg_ptrs...), C_NULL, C_NULL), num_groups, exception_info)
            else
                host_dispatch(h, ($(arg_ptrs...),), num_groups, exception_info)
            end
        end
        nothing
    end
end
