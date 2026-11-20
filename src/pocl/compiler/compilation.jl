## gpucompiler interface

Base.@kwdef struct OpenCLCompilerParams <: AbstractCompilerParams
    # request a fixed sub-group width via `intel_reqd_sub_group_size`
    sub_group_size::Union{Nothing, Int} = nothing
    # lower `muladd` to `fma` (see `muladd_method_table`)
    fuse_muladd::Bool = false
end

const OpenCLCompilerConfig = CompilerConfig{SPIRVCompilerTarget, OpenCLCompilerParams}
const OpenCLCompilerJob = CompilerJob{SPIRVCompilerTarget, OpenCLCompilerParams}

"""
    OpenCLResults

Cached compilation results for an OpenCL kernel job, managed by
`GPUCompiler.cached_results`. Fields are populated through the compile pipeline:
`obj` (SPIR-V bytes) + `entry` + `device_rng` + `device_heap` after codegen, and `kernels` after the
session-local link onto an OpenCL context. The first three are session-portable
(cached through precompilation, except when GPUCompiler marks the job
session-dependent and wipes its entries before image serialization); `kernels` is
session-local and never populated during precompilation. `obj === nothing`
identifies a job that has not been compiled yet.

`kernels` holds the `cl.Kernel` linked on the session's context, paired with that context.
The cache partition already covers everything that affects codegen via
`GPUCompiler.cache_owner`, so the only runtime-visible dimension left is the OpenCL context
that owns the linked `cl.Kernel`. There's one context per session, so this holds at most
one entry; the context identifies kernels from before a reset of the session.
"""
mutable struct OpenCLResults
    obj::Union{Nothing, Vector{UInt8}}                   # SPIR-V binary
    entry::Union{Nothing, String}
    device_rng::Bool
    device_heap::Bool
    exception_sites::Vector{GPUCompiler.ExceptionSite}
    kernels::Vector{Tuple{cl.Context, cl.Kernel}}        # session-local; linear-scanned
    OpenCLResults() = new(
        nothing, nothing, false, false, GPUCompiler.ExceptionSite[], Tuple{cl.Context, cl.Kernel}[]
    )
end

GPUCompiler.runtime_module(::CompilerJob{<:Any, OpenCLCompilerParams}) = POCL
GPUCompiler.uses_exception_sites(::CompilerJob{<:Any, OpenCLCompilerParams}) = true

# SPIR-V doesn't support 128-bit integers, so keep 64-bit integer math (e.g. `Rational`
# comparisons) that Base computes with them out of Int128. Some of Base's error paths format
# their message at run time, which doesn't compile; GPUToolbox's overrides throw with a
# constant message instead.
GPUCompiler.method_tables(job::OpenCLCompilerJob) =
    job.config.params.fuse_muladd ?
    (muladd_method_table, method_table, SPIRVIntrinsics.method_table, GPUToolbox.Overlays.int128_overrides, GPUToolbox.Overlays.throw_overrides) :
    (method_table, SPIRVIntrinsics.method_table, GPUToolbox.Overlays.int128_overrides, GPUToolbox.Overlays.throw_overrides)

# filter out OpenCL built-ins
# TODO: eagerly lower these using the translator API
GPUCompiler.isintrinsic(job::OpenCLCompilerJob, fn::String) =
    invoke(
    GPUCompiler.isintrinsic,
    Tuple{CompilerJob{SPIRVCompilerTarget}, typeof(fn)},
    job, fn
) ||
    in(fn, known_intrinsics) ||
    contains(fn, "__spirv_")

GPUCompiler.kernel_state_type(::OpenCLCompilerJob) = KernelState

function GPUCompiler.finish_module!(
        @nospecialize(job::OpenCLCompilerJob),
        mod::LLVM.Module, entry::LLVM.Function
    )
    entry = invoke(
        GPUCompiler.finish_module!,
        Tuple{CompilerJob{SPIRVCompilerTarget}, LLVM.Module, LLVM.Function},
        job, mod, entry
    )

    # the built-ins of SPIRVIntrinsics don't free memory. Enzyme needs to know that to
    # differentiate callers in reverse mode, and cannot tell from a declaration.
    # XXX: SPIRVIntrinsics should declare them `nofree`
    for f in mod.functions
        if isempty(f.blocks) && contains(f.name, "__spirv_BuiltIn")
            push!(f.function_attributes, EnumAttribute(:nofree))
        end
    end

    sg_size = job.config.params.sub_group_size
    if sg_size !== nothing
        entry.metadata["intel_reqd_sub_group_size"] = MDNode([ConstantInt(Int32(sg_size))])
    end

    # if this kernel uses our RNG, we should prime the shared state.
    # XXX: these transformations should really happen at the Julia IR level...
    if haskey(mod.functions, "julia.opencl.random_keys") && job.config.kernel
        # insert call to `initialize_rng_state`
        f = initialize_rng_state
        ft = typeof(f)
        tt = Tuple{}

        # create a deferred compilation job for `initialize_rng_state`. it is an overlay, so
        # look it up in the job's method tables.
        src = GPUCompiler.generic_methodinstance(
            ft, tt, job.world; method_table_view = GPUCompiler.method_table_view(job)
        )
        cfg = CompilerConfig(job.config; kernel = false, name = nothing)
        job = CompilerJob(src, cfg, job.world)
        id = length(GPUCompiler.deferred_codegen_jobs) + 1
        GPUCompiler.deferred_codegen_jobs[id] = job

        # generate IR for calls to `deferred_codegen` and the resulting function pointer
        top_bb = entry.entry
        bb = BasicBlock(LLVM.before(top_bb), "initialize_rng")
        @dispose builder = IRBuilder() begin
            position!(builder, LLVM.at_end(bb))
            subprogram = entry.subprogram
            if subprogram !== nothing
                loc = DILocation(0, 0, subprogram)
                builder.debug_location = loc
            end

            # call the `deferred_codegen` marker function
            # (declared like GPUCompiler's `ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, ...)`)
            T_ptr = convert(LLVMType, Ptr{Cvoid})
            T_id = convert(LLVMType, Int)
            deferred_codegen_ft = LLVM.FunctionType(T_ptr, [T_id])
            deferred_codegen = get!(mod.functions, "deferred_codegen") do
                LLVM.Function(mod, "deferred_codegen", deferred_codegen_ft)
            end
            fptr = call!(builder, deferred_codegen_ft, deferred_codegen, [ConstantInt(id)])

            # call the `initialize_rng_state` function, which returns nothing. inferring its
            # return type natively would leave host code for it in sysimages.
            llvm_ft = LLVM.FunctionType(LLVM.VoidType())
            fptr = inttoptr!(builder, fptr, LLVM.PointerType(llvm_ft))
            call!(builder, llvm_ft, fptr)
            br!(builder, top_bb)

            # note the use of the device-side RNG in this kernel
            push!(entry.function_attributes, StringAttribute("julia.opencl.rng", ""))
        end

        # XXX: put some of the above behind GPUCompiler abstractions
        #      (e.g., a compile-time version of `deferred_codegen`)
    end
    return entry
end

# Enzyme caches values of loops with dynamic trip counts in memory it allocates with libc's
# `malloc` and grows with `realloc`, which device code cannot call. allocate from the launch's
# heap instead (see `malloc`), by calling the runtime's `malloc`: this runs before the runtime
# is linked, which then provides it. blocks are released with the heap, not by `free`.
function GPUCompiler.link_libraries!(@nospecialize(job::OpenCLCompilerJob), mod::LLVM.Module)
    replace_allocator!(job, mod, "malloc", pocl_malloc!)
    replace_allocator!(job, mod, "realloc", pocl_realloc!)
    replace_allocator!(job, mod, "free", (job, mod, builder, f) -> ret!(builder))
    return
end

# replace the declaration `name` with an internal definition, whose body `body!` emits
function replace_allocator!(@nospecialize(job::CompilerJob), mod::LLVM.Module, name, body!)
    decl = get(mod.functions, name, nothing)
    (decl === nothing || !LLVM.isdeclaration(decl)) && return
    f = LLVM.Function(mod, "pocl_$name", LLVM.function_type(decl))
    LLVM.linkage!(f, LLVM.API.LLVMInternalLinkage)
    push!(f.function_attributes, EnumAttribute("alwaysinline"))
    @dispose builder = IRBuilder() begin
        position!(builder, LLVM.at_end(BasicBlock(f, "entry")))
        body!(job, mod, builder, f)
    end
    LLVM.replace_uses!(decl, f)
    LLVM.erase!(decl)
    return
end

# a block of the launch's heap, of the size that is the last parameter of `f`
function heap_block!(@nospecialize(job::CompilerJob), builder::IRBuilder, f::LLVM.Function)
    T_ptr = LLVM.return_type(LLVM.function_type(f))
    block = call!(builder, GPUCompiler.Runtime.get(:malloc), [last(LLVM.parameters(f))]; job)
    # (pointers are integers on Julia 1.11 and earlier)
    return LLVM.value_type(block) == T_ptr ? block : inttoptr!(builder, block, T_ptr)
end

pocl_malloc!(job, mod, builder, f) = ret!(builder, heap_block!(job, builder, f))

# blocks are preceded by their size (see `malloc`)
function pocl_realloc!(@nospecialize(job::CompilerJob), mod::LLVM.Module, builder::IRBuilder, f::LLVM.Function)
    old, sz = LLVM.parameters(f)
    T_int = LLVM.value_type(sz)
    new = heap_block!(job, builder, f)
    copy = BasicBlock(f, "copy")
    done = BasicBlock(f, "done")
    br!(builder, icmp!(builder, LLVM.API.LLVMIntEQ, old, LLVM.null(LLVM.value_type(old))), done, copy)

    position!(builder, LLVM.at_end(copy))
    header = gep!(builder, LLVM.Int8Type(), old, [ConstantInt(T_int, -heap_alignment)])
    old_sz = load!(builder, T_int, header)
    len = select!(builder, icmp!(builder, LLVM.API.LLVMIntULT, old_sz, sz), old_sz, sz)
    memcpy!(builder, new, heap_alignment, old, heap_alignment, len)
    br!(builder, done)

    position!(builder, LLVM.at_end(done))
    ret!(builder, new)
    return
end

function GPUCompiler.finish_linked_module!(@nospecialize(job::OpenCLCompilerJob), mod::LLVM.Module)
    # note the use of the heap (see `malloc`), which the runtime is linked for
    if haskey(mod.functions, GPUCompiler.Runtime.get(:malloc).llvm_name) && job.config.kernel
        for f in GPUCompiler.kernels(mod)
            push!(f.function_attributes, StringAttribute("julia.opencl.heap", ""))
        end
    end

    for f in GPUCompiler.kernels(mod)
        kernel_intrinsics = Dict(
            "julia.opencl.random_keys" => (; name = "random_keys", typ = LLVMPtr{UInt32, AS.Workgroup}),
            "julia.opencl.random_counters" => (; name = "random_counters", typ = LLVMPtr{UInt32, AS.Workgroup}),
        )
        GPUCompiler.add_input_arguments!(job, mod, f, kernel_intrinsics)
    end
    return
end


## compiler implementation (configure, compile, and link)

"""
    spirv_atomics(dev)

The atomic operations `dev` supports, for which `SPIRVCompilerTarget` selects SPIR-V
instructions instead of compare-and-swap loops.

Floating-point addition is supported per precision and address space as `dev` reports it
through `cl_ext_float_atomics`, for half and double precision only if `dev` supports those
types. 64-bit integer atomics need both `cl_khr_int64_base_atomics` and
`cl_khr_int64_extended_atomics`.
"""
function spirv_atomics(dev)
    exts = dev.extensions
    f16 = "cl_khr_fp16" in exts ? dev.half_fp_atomic_capabilities : zero(UInt64)
    f32 = dev.single_fp_atomic_capabilities
    f64 = "cl_khr_fp64" in exts ? dev.double_fp_atomic_capabilities : zero(UInt64)
    global_add(caps) = caps & cl.CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT != 0
    local_add(caps) = caps & cl.CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT != 0
    return SPIRVAtomics(;
        int64 = "cl_khr_int64_base_atomics" in exts && "cl_khr_int64_extended_atomics" in exts,
        fadd_f16_global = global_add(f16), fadd_f16_local = local_add(f16),
        fadd_f32_global = global_add(f32), fadd_f32_local = local_add(f32),
        fadd_f64_global = global_add(f64), fadd_f64_local = local_add(f64),
    )
end

"""
    compiler_config(dev; kwargs...)

The GPUCompiler configuration for compiling kernels for `dev`, cached per device and
keyword arguments. Besides those of `CompilerConfig` (`kernel`, `name`, `always_inline`,
`debug_level`) and `sub_group_size` (the sub-group width to require, or `nothing` for none,
defaulting to the width `KI.sub_group_size` reports), it takes:

- `atomics`: override the atomic capabilities GPUCompiler may select directly. This
  replaces the whole device-derived `SPIRVAtomics` (see `spirv_atomics`), and defaults to
  the device's capabilities. Enabling capabilities the device doesn't support can make
  compilation fail or crash the driver's compiler, while disabling them relies on integer
  compare-and-swap for the fallback.
- `extensions`: SPIR-V extensions to enable, as a `--spirv-ext` specifier (e.g.
  `"+SPV_KHR_expect_assume"`), in addition to those the atomics need, to
  `SPV_KHR_no_integer_wrap_decoration`, and to `SPV_INTEL_subgroups` on devices with
  `cl_intel_subgroups` (for the vector shuffles of packed structs). `nothing`, the default,
  enables no others.

Other keyword arguments are passed on to `SPIRVCompilerTarget`.
"""
function compiler_config end

# `SPV_KHR_no_integer_wrap_decoration` carries the `nuw`/`nsw` flags of integer arithmetic,
# e.g. of the index computations, over to PoCL (#868)
const default_extensions = "+SPV_KHR_no_integer_wrap_decoration"

# cache of compiler configurations, per device (but additionally configurable via kwargs)
const _toolchain = Ref{Any}()
const _compiler_configs = Dict{UInt, OpenCLCompilerConfig}()
function compiler_config(dev::cl.Device; kwargs...)
    # (as a `NamedTuple`: hashing the `Pairs` boxes each value, on every launch)
    h = hash(dev, hash(values(kwargs)))
    # launches already hold this (reentrant) lock, but reflection doesn't
    return @lock clfunction_lock begin
        config = get(_compiler_configs, h, nothing)
        if config === nothing
            config = _compiler_config(dev; kwargs...)
            _compiler_configs[h] = config
        end
        config
    end
end
@noinline function _compiler_config(
        dev; kernel = true, name = nothing, always_inline = false,
        debug_level = Base.JLOptions().debug_level,
        sub_group_size::Union{Nothing, Int} = default_sub_group_size(dev),
        atomics::SPIRVAtomics = spirv_atomics(dev),
        extensions::Union{Nothing, String} = nothing, kwargs...
    )
    supports_fp16 = "cl_khr_fp16" in dev.extensions
    supports_fp64 = "cl_khr_fp64" in dev.extensions
    # `BFloat16` loads, stores and conversions. GPUCompiler widens the arithmetic to `float`,
    # because PoCL doesn't accept any SPIR-V extension for bfloat16 arithmetic.
    supports_bfloat16 = "SPV_KHR_bfloat16" in dev.spirv_extensions

    if sub_group_size !== nothing && sub_group_size ∉ dev.sub_group_sizes
        error("$sub_group_size is not a valid sub-group size for this device.")
    end

    if supports_bfloat16
        extensions = extensions === nothing ? "+SPV_KHR_bfloat16" : "+SPV_KHR_bfloat16,$extensions"
    end

    # the vector shuffles of `KI.shfl_word_tuple` (see `POCLKernels`); `SPV_INTEL_subgroups`
    # is only declared by modules that use it
    if "cl_intel_subgroups" in dev.extensions
        extensions = extensions === nothing ? "+SPV_INTEL_subgroups" : "+SPV_INTEL_subgroups,$extensions"
    end

    # create GPUCompiler objects
    target = SPIRVCompilerTarget(;
        supports_fp16, supports_fp64, supports_bfloat16, atomics,
        extensions = join(filter(!isempty, [default_extensions, something(extensions, "")]), ","),
        validate = true, kwargs...
    )
    params = OpenCLCompilerParams(; sub_group_size, fuse_muladd = fast_fma())
    return CompilerConfig(target, params; kernel, name, always_inline, debug_level)
end

# whether the CPU that PoCL runs kernels on, i.e., the host, executes `fma` in hardware, as
# fast as a multiplication or addition. GPUCompiler's `have_fma` doesn't tell: on SPIR-V, it
# holds whenever OpenCL's (correctly rounded) `fma` is available, possibly in software.
function fast_fma()
    Sys.ARCH === :aarch64 && return true
    Sys.ARCH in (:x86_64, :i686) || return false
    # LLVM's features of the host CPU, which PoCL compiles for
    features = ccall(:jl_get_cpu_features, String, ())
    return "+fma" in split(features, ',')
end

# The world in which this package was loaded. Running the compiler in that world reuses the
# native code that precompilation generated for it, even when packages loaded afterwards
# invalidate some of it (e.g. by adding methods to Base functions the compiler calls).
# Kernels themselves are still compiled for the current world (`job.world`), but methods of
# the compiler's interface (e.g. `GPUCompiler.finish_module!`) that are added after loading,
# for example by Revise, aren't used. Before `__init__` runs, as during precompilation,
# `invoke_in_world` clamps the world to the current one.
const initialization_world = Ref{UInt}(typemax(UInt))

invoke_frozen(f, args...) = Base.invoke_in_world(initialization_world[], f, args...)

# run inference + LLVM codegen + SPIR-V emission. returns `(obj, entry, device_rng,
# device_heap, exception_sites)`,
# all session-portable so they survive precompilation when stored on a cached `CodeInstance`.
const compilations = Threads.Atomic{Int}(0)
function compile_to_obj(@nospecialize(job::CompilerJob))
    compilations[] += 1

    return JuliaContext() do ctx
        obj, meta = invoke_frozen(GPUCompiler.compile, :obj, job)

        # we own the IR: inspect it, then dispose of it
        @dispose ir = meta.ir begin
            entry = meta.entry.name
            device_rng = haskey(meta.entry.function_attributes, "julia.opencl.rng")
            device_heap = haskey(meta.entry.function_attributes, "julia.opencl.heap")
            (; obj, entry, device_rng, device_heap, meta.exception_sites)
        end
    end
end

# link the SPIR-V bytes into a session-local `cl.Kernel` on the active context.
function link_kernel(
        @nospecialize(job::CompilerJob), obj::Vector{UInt8}, entry::String,
        exception_sites::Vector{GPUCompiler.ExceptionSite}, device_heap::Bool
    )
    prog = if "cl_khr_il_program" in device().extensions
        cl.Program(obj, context())
    else
        error("Your device does not support SPIR-V, which is currently required for native execution.")
    end
    cl.build!(prog)
    kernel = cl.Kernel(prog, entry)
    # cap tuned work-groups to avoid the queue fallback for excessive stack use; fitting to
    # the ndrange can only shrink them further
    kernel.tuned_work_group_size = min(kernel.work_group_size, host_kernel_max_items(kernel))
    kernel.exception_sites = exception_sites
    kernel.device_heap = device_heap
    return kernel
end
