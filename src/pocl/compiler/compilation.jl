## gpucompiler interface

Base.@kwdef struct OpenCLCompilerParams <: AbstractCompilerParams
    # request a fixed sub-group width via `intel_reqd_sub_group_size`
    sub_group_size::Union{Nothing, Int} = nothing
end

const OpenCLCompilerConfig = CompilerConfig{SPIRVCompilerTarget, OpenCLCompilerParams}
const OpenCLCompilerJob = CompilerJob{SPIRVCompilerTarget, OpenCLCompilerParams}

"""
    OpenCLResults

Cached compilation results for an OpenCL kernel job, managed by
`GPUCompiler.cached_results`. Fields are populated through the compile pipeline:
`obj` (SPIR-V bytes) + `entry` + `device_rng` after codegen, and `kernels` after the
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
    kernels::Vector{Tuple{cl.Context, cl.Kernel}}        # session-local; linear-scanned
    OpenCLResults() = new(nothing, nothing, false, Tuple{cl.Context, cl.Kernel}[])
end

GPUCompiler.runtime_module(::CompilerJob{<:Any, OpenCLCompilerParams}) = POCL

GPUCompiler.method_tables(::OpenCLCompilerJob) = (method_table, SPIRVIntrinsics.method_table)

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

        # create a deferred compilation job for `initialize_rng_state`
        src = methodinstance(ft, tt, job.world)
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

            # call the `initialize_rng_state` function
            rt = Core.Compiler.return_type(f, tt)
            llvm_rt = convert(LLVMType, rt)
            llvm_ft = LLVM.FunctionType(llvm_rt)
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

function GPUCompiler.finish_linked_module!(@nospecialize(job::OpenCLCompilerJob), mod::LLVM.Module)
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
`debug_level`) and `sub_group_size`, it takes:

- `atomics`: override the atomic capabilities GPUCompiler may select directly. This
  replaces the whole device-derived `SPIRVAtomics` (see `spirv_atomics`), and defaults to
  the device's capabilities. Enabling capabilities the device doesn't support can make
  compilation fail or crash the driver's compiler, while disabling them relies on integer
  compare-and-swap for the fallback.
- `extensions`: SPIR-V extensions to enable, as a `--spirv-ext` specifier (e.g.
  `"+SPV_KHR_expect_assume"`), in addition to those the atomics need. `nothing`, the
  default, enables no others.

Other keyword arguments are passed on to `SPIRVCompilerTarget`.
"""
function compiler_config end

# cache of compiler configurations, per device (but additionally configurable via kwargs)
const _toolchain = Ref{Any}()
const _compiler_configs = Dict{UInt, OpenCLCompilerConfig}()
function compiler_config(dev::cl.Device; kwargs...)
    h = hash(dev, hash(kwargs))
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
        sub_group_size::Union{Nothing, Int} = 32,
        atomics::SPIRVAtomics = spirv_atomics(dev),
        extensions::Union{Nothing, String} = nothing, kwargs...
    )
    supports_fp16 = "cl_khr_fp16" in dev.extensions
    supports_fp64 = "cl_khr_fp64" in dev.extensions

    if sub_group_size !== nothing && sub_group_size ∉ dev.sub_group_sizes
        error("$sub_group_size is not a valid sub-group size for this device.")
    end

    # create GPUCompiler objects
    target = SPIRVCompilerTarget(;
        supports_fp16, supports_fp64, atomics, extensions = something(extensions, ""),
        validate = true, kwargs...
    )
    params = OpenCLCompilerParams(; sub_group_size)
    return CompilerConfig(target, params; kernel, name, always_inline, debug_level)
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

# run inference + LLVM codegen + SPIR-V emission. returns `(obj, entry, device_rng)`,
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
            (; obj, entry, device_rng)
        end
    end
end

# link the SPIR-V bytes into a session-local `cl.Kernel` on the active context.
function link_kernel(@nospecialize(job::CompilerJob), obj::Vector{UInt8}, entry::String)
    prog = if "cl_khr_il_program" in device().extensions
        cl.Program(obj, context())
    else
        error("Your device does not support SPIR-V, which is currently required for native execution.")
    end
    cl.build!(prog)
    return cl.Kernel(prog, entry)
end
