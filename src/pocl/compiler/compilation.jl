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

`kernels` is a small linear cache of `(cl.Context, cl.Kernel)` pairs. The cache partition
already covers everything that affects codegen via `GPUCompiler.cache_owner`, so the only
runtime-visible dimension left is the OpenCL context that owns the linked `cl.Kernel`.
A linear scan with `===` is fastest in the common case (n=1) and stays cheap for the
rare workload that bounces between a handful of contexts on the same device.
"""
mutable struct OpenCLResults
    obj::Union{Nothing, Vector{UInt8}}                   # SPIR-V binary
    entry::Union{Nothing, String}
    device_rng::Bool
    kernels::Vector{Tuple{cl.Context, cl.Kernel}}        # session-local; linear-scanned
    OpenCLResults() = new(nothing, nothing, false, Tuple{cl.Context, cl.Kernel}[])
end

GPUCompiler.runtime_module(::CompilerJob{<:Any, OpenCLCompilerParams}) = POCL

GPUCompiler.method_table_view(job::OpenCLCompilerJob) = GPUCompiler.StackedMethodTable(job.world, method_table, SPIRVIntrinsics.method_table)

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
        metadata(entry)["intel_reqd_sub_group_size"] = MDNode([ConstantInt(Int32(sg_size))])
    end

    # if this kernel uses our RNG, we should prime the shared state.
    # XXX: these transformations should really happen at the Julia IR level...
    if haskey(functions(mod), "julia.opencl.random_keys") && job.config.kernel
        # insert call to `initialize_rng_state`
        f = initialize_rng_state
        ft = typeof(f)
        tt = Tuple{}

        # create a deferred compilation job for `initialize_rng_state`
        src = methodinstance(ft, tt, GPUCompiler.tls_world_age())
        cfg = CompilerConfig(job.config; kernel = false, name = nothing)
        job = CompilerJob(src, cfg, job.world)
        id = length(GPUCompiler.deferred_codegen_jobs) + 1
        GPUCompiler.deferred_codegen_jobs[id] = job

        # generate IR for calls to `deferred_codegen` and the resulting function pointer
        top_bb = first(blocks(entry))
        bb = BasicBlock(top_bb, "initialize_rng")
        @dispose builder = IRBuilder() begin
            position!(builder, bb)
            subprogram = LLVM.subprogram(entry)
            if subprogram !== nothing
                loc = DILocation(0, 0, subprogram)
                debuglocation!(builder, loc)
            end
            debuglocation!(builder, first(instructions(top_bb)))

            # call the `deferred_codegen` marker function
            T_ptr = if LLVM.version() >= v"17"
                LLVM.PointerType()
            elseif VERSION >= v"1.12.0-DEV.225"
                LLVM.PointerType(LLVM.Int8Type())
            else
                LLVM.Int64Type()
            end
            T_id = convert(LLVMType, Int)
            deferred_codegen_ft = LLVM.FunctionType(T_ptr, [T_id])
            deferred_codegen = if haskey(functions(mod), "deferred_codegen")
                functions(mod)["deferred_codegen"]
            else
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
            push!(function_attributes(entry), StringAttribute("julia.opencl.rng", ""))
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

function GPUCompiler.finish_ir!(
        @nospecialize(job::OpenCLCompilerJob),
        mod::LLVM.Module, entry::LLVM.Function
    )
    entry = invoke(
        GPUCompiler.finish_ir!,
        Tuple{CompilerJob{SPIRVCompilerTarget}, LLVM.Module, LLVM.Function},
        job, mod, entry
    )

    # Deferred-codegen entrypoints -- notably the wrappers Enzyme generates -- are
    # held externally live across `InternalizePass` so that linking can resolve
    # them. Once linked and `alwaysinline`d they are dead, but external linkage
    # keeps `GlobalDCEPass` from dropping them, so the SPIR-V backend still has to
    # translate a function it cannot express: they take first-class aggregates
    # containing `addrspace(1)` pointers, and extracting one back out miscompiles
    # (the backend types the struct member as a pointer-to-uchar but the extract
    # result as pointer-to-double, which fails `spirv-val`).
    #
    # Drop the bodies of unreferenced non-kernel definitions so only declarations
    # remain. We empty rather than erase so the `finish_ir!` loop over the
    # remaining deferred jobs can still look them up by name.
    if job.config.kernel
        for f in functions(mod)
            f == entry && continue
            isdeclaration(f) && continue
            LLVM.isintrinsic(f) && continue
            LLVM.callconv(f) == LLVM.API.LLVMSPIRKERNELCallConv && continue
            isempty(uses(f)) || continue
            empty!(f)
        end
    end

    return entry
end


## compiler implementation (configure, compile, and link)

# cache of compiler configurations, per device (but additionally configurable via kwargs)
const _toolchain = Ref{Any}()
const _compiler_configs = Dict{UInt, OpenCLCompilerConfig}()
function compiler_config(dev::cl.Device; kwargs...)
    h = hash(dev, hash(kwargs))
    config = get(_compiler_configs, h, nothing)
    if config === nothing
        config = _compiler_config(dev; kwargs...)
        _compiler_configs[h] = config
    end
    return config
end
@noinline function _compiler_config(dev; kernel = true, name = nothing, always_inline = false, sub_group_size::Union{Nothing, Int} = 32, kwargs...)
    supports_fp16 = "cl_khr_fp16" in dev.extensions
    supports_fp64 = "cl_khr_fp64" in dev.extensions

    if sub_group_size !== nothing && sub_group_size ∉ dev.sub_group_sizes
        error("$sub_group_size is not a valid sub-group size for this device.")
    end

    # create GPUCompiler objects
    target = SPIRVCompilerTarget(; supports_fp16, supports_fp64, validate = true, kwargs...)
    params = OpenCLCompilerParams(; sub_group_size)
    return CompilerConfig(target, params; kernel, name, always_inline)
end

# run inference + LLVM codegen + SPIR-V emission. returns `(obj, entry, device_rng)`,
# all session-portable so they survive precompilation when stored on a cached `CodeInstance`.
const compilations = Threads.Atomic{Int}(0)
function compile_to_obj(@nospecialize(job::CompilerJob))
    compilations[] += 1

    return JuliaContext() do ctx
        obj, meta = GPUCompiler.compile(:obj, job)

        entry = LLVM.name(meta.entry)
        device_rng = StringAttribute("julia.opencl.rng", "") in collect(function_attributes(meta.entry))

        (; obj, entry, device_rng)
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
