## gpucompiler interface

struct OpenCLCompilerParams <: AbstractCompilerParams end
const OpenCLCompilerConfig = CompilerConfig{SPIRVCompilerTarget, OpenCLCompilerParams}
const OpenCLCompilerJob = CompilerJob{SPIRVCompilerTarget, OpenCLCompilerParams}

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

function GPUCompiler.finish_module!(@nospecialize(job::OpenCLCompilerJob),
                                          mod::LLVM.Module, entry::LLVM.Function)
    entry = invoke(GPUCompiler.finish_module!,
                   Tuple{CompilerJob{SPIRVCompilerTarget}, LLVM.Module, LLVM.Function},
                   job, mod, entry)

    # if this kernel uses our RNG, we should prime the shared state.
    # XXX: these transformations should really happen at the Julia IR level...
    if haskey(functions(mod), "julia.opencl.random_keys") && job.config.kernel
        # insert call to `initialize_rng_state`
        f = initialize_rng_state
        ft = typeof(f)
        tt = Tuple{}

        # create a deferred compilation job for `initialize_rng_state`
        src = methodinstance(ft, tt, GPUCompiler.tls_world_age())
        cfg = CompilerConfig(job.config; kernel=false, name=nothing)
        job = CompilerJob(src, cfg, job.world)
        id = length(GPUCompiler.deferred_codegen_jobs) + 1
        GPUCompiler.deferred_codegen_jobs[id] = job

        # generate IR for calls to `deferred_codegen` and the resulting function pointer
        top_bb = first(blocks(entry))
        bb = BasicBlock(top_bb, "initialize_rng")
        @dispose builder=IRBuilder() begin
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


## compiler implementation (cache, configure, compile, and link)

# cache of compilation caches, per context
const _compiler_caches = Dict{cl.Context, Dict{Any, Any}}()
function compiler_cache(ctx::cl.Context)
    cache = get(_compiler_caches, ctx, nothing)
    if cache === nothing
        cache = Dict{Any, Any}()
        _compiler_caches[ctx] = cache
    end
    return cache
end

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
@noinline function _compiler_config(dev; kernel = true, name = nothing, always_inline = false, kwargs...)
    supports_fp16 = "cl_khr_fp16" in dev.extensions
    supports_fp64 = "cl_khr_fp64" in dev.extensions


    # create GPUCompiler objects
    target = SPIRVCompilerTarget(; supports_fp16, supports_fp64, kwargs...)
    params = OpenCLCompilerParams()
    return CompilerConfig(target, params; kernel, name, always_inline)
end

# compile to executable machine code
function compile(@nospecialize(job::CompilerJob))
    # TODO: this creates a context; cache those.
    obj, meta = JuliaContext() do ctx
        obj, meta = GPUCompiler.compile(:obj, job)

        entry = LLVM.name(meta.entry)
        device_rng = StringAttribute("julia.opencl.rng", "") in collect(function_attributes(meta.entry))

        (; obj, entry, device_rng)
    end
end

# link into an executable kernel
function link(@nospecialize(job::CompilerJob), compiled)
    prog = if "cl_khr_il_program" in device().extensions
        cl.Program(compiled.obj, context())
    else
        error("Your device does not support SPIR-V, which is currently required for native execution.")
    end
    cl.build!(prog)
    (; kernel=cl.Kernel(prog, compiled.entry), compiled.device_rng)
end
