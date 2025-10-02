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
        GPUCompiler.compile(:obj, job)
    end

    return (; obj, entry = LLVM.name(meta.entry))
end

# link into an executable kernel
function link(@nospecialize(job::CompilerJob), compiled)
    prog = if "cl_khr_il_program" in device().extensions
        cl.Program(compiled.obj, context())
    else
        error("Your device does not support SPIR-V, which is currently required for native execution.")
    end
    cl.build!(prog)
    return cl.Kernel(prog, compiled.entry)
end
