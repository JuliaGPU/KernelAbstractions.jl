using KernelAbstractions
using Test
import KernelAbstractions.POCL: cl

@testset "PoCL functionality" begin
    function default_pocl_device()
        cl.pocl_standalone_jll.is_available() || return nothing
        platforms = cl.platforms()
        idx = findfirst(p -> p.vendor == "The pocl project", platforms)
        return isnothing(idx) ? nothing : cl.default_device(platforms[idx])
    end
    function device_accepts_kernels(device)
        device.available && "cl_pocl_host_kernel" in device.extensions || return false
        size = Ref{Csize_t}()
        cl.clGetDeviceInfo(device, cl.CL_DEVICE_IL_VERSION, 0, C_NULL, size)
        iszero(size[]) && return false
        version = zeros(UInt8, size[])
        cl.clGetDeviceInfo(device, cl.CL_DEVICE_IL_VERSION, length(version), version, C_NULL)
        return any(startswith("SPIR-V_"), split(chop(String(version); tail = 1)))
    end
    device = default_pocl_device()
    threads = cl.cpu_threads()
    expected = if isnothing(device) || !device.available
        false
    elseif threads !== nothing && threads < device.max_compute_units && device.max_sub_devices > 0
        sub = cl.sub_device(device, threads)
        try
            device_accepts_kernels(sub)
        finally
            @test ccall((:POclReleaseDevice, cl.libopencl), Cint, (cl.cl_device_id,), sub) == cl.CL_SUCCESS
        end
    else
        device_accepts_kernels(device)
    end
    @test KernelAbstractions.functional(CPU()) === expected

    if device !== nothing && device.available && device.max_sub_devices > 0
        function refcount(d)
            count = Ref{Cuint}()
            cl.clGetDeviceInfo(d, cl.CL_DEVICE_REFERENCE_COUNT, sizeof(Cuint), count, C_NULL)
            return count[]
        end
        for _ in 1:2
            context, selected, count = Ref{cl.Context}(), Ref{cl.Device}(), Ref{Cuint}()
            @test_throws cl.CLError(cl.CL_DEVICE_NOT_AVAILABLE) cl.with_sub_device(device, 1) do sub
                context[] = cl.Context(sub)
                selected[] = sub
                count[] = refcount(sub)
                throw(cl.CLError(cl.CL_DEVICE_NOT_AVAILABLE))
            end
            @test refcount(selected[]) == count[] - 1
            finalize(context[])
        end
    end

    julia = Cmd(filter(arg -> !startswith(arg, "--code-coverage"), Base.julia_cmd().exec))
    script = """
    using KernelAbstractions, Test
    @kernel function fill_one!(a)
        a[@index(Global, Linear)] = 1
    end
    @test !KernelAbstractions.functional(CPU())
    @test_throws r"PoCL has no default device|POCL not available|The CPU back-end needs PoCL" fill_one!(CPU())(zeros(Int32, 1); ndrange = 1)

    import KernelAbstractions.POCL
    POCL.reset_session_state!()
    const device_error = Ref(POCL.cl.CL_DEVICE_NOT_AVAILABLE)
    @eval POCL try_init_session() = throw(cl.CLError(Main.device_error[]))
    @test !KernelAbstractions.functional(CPU())
    @test_throws POCL.cl.CLError(device_error[]) fill_one!(CPU())(zeros(Int32, 1); ndrange = 1)
    device_error[] = POCL.cl.CL_DEVICE_NOT_FOUND
    @test_throws POCL.cl.CLError(device_error[]) KernelAbstractions.functional(CPU())
    """
    cmd = `$julia --startup-file=no --project=$(Base.active_project()) -e $script`
    child = addenv(cmd, "POCL_DEVICES" => "nonexistent")
    @test success(pipeline(child; stdout, stderr))
end
