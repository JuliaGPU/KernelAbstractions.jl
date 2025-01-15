module nanoOpenCL

using ..POCL: platform, device, context, queue

import OpenCL_jll
import pocl_jll

using Printf

const libopencl = OpenCL_jll.libopencl # TODO directly use POCL

"""
    @checked function foo(...)
        rv = ...
        return rv
    end

Macro for wrapping a function definition returning a status code. Two versions of the
function will be generated: `foo`, with the function body wrapped by an invocation of the
`check` function (to be implemented by the caller of this macro), and `unchecked_foo` where no
such invocation is present and the status code is returned to the caller.
"""
macro checked(ex)
    # parse the function definition
    @assert Meta.isexpr(ex, :function)
    sig = ex.args[1]
    @assert Meta.isexpr(sig, :call)
    body = ex.args[2]
    @assert Meta.isexpr(body, :block)

    # we need to detect the first API call, so add an initialization check
    body = quote
        if !initialized[]
            initialize()
        end
        $body
    end

    # generate a "safe" version that performs a check
    safe_body = quote
        check() do
            $body
        end
    end
    safe_sig = Expr(:call, sig.args[1], sig.args[2:end]...)
    safe_def = Expr(:function, safe_sig, safe_body)

    # generate a "unchecked" version that returns the error code instead
    unchecked_sig = Expr(:call, Symbol("unchecked_", sig.args[1]), sig.args[2:end]...)
    unchecked_def = Expr(:function, unchecked_sig, body)

    return esc(:($safe_def, $unchecked_def))
end

const CL_SUCCESS = 0

const CL_DEVICE_NOT_FOUND = -1

const CL_DEVICE_NOT_AVAILABLE = -2

const CL_INVALID_ARG_INDEX = -49

const CL_INVALID_ARG_VALUE = -50

const CL_INVALID_ARG_SIZE = -51

const CL_INVALID_KERNEL_ARGS = -52

const CL_PLATFORM_NOT_FOUND_KHR = -1001

const CL_PLATFORM_PROFILE = 0x0900

const CL_PLATFORM_VERSION = 0x0901

const CL_PLATFORM_NAME = 0x0902

const CL_PLATFORM_VENDOR = 0x0903

const CL_PLATFORM_EXTENSIONS = 0x0904

const CL_PLATFORM_HOST_TIMER_RESOLUTION = 0x0905

const CL_PLATFORM_NUMERIC_VERSION = 0x0906

const CL_PLATFORM_EXTENSIONS_WITH_VERSION = 0x0907

const CL_DEVICE_TYPE_DEFAULT = 1 << 0

const CL_DEVICE_TYPE_CPU = 1 << 1

const CL_DEVICE_TYPE_GPU = 1 << 2

const CL_DEVICE_TYPE_ACCELERATOR = 1 << 3

const CL_DEVICE_TYPE_CUSTOM = 1 << 4

const CL_DEVICE_TYPE_ALL = 0xffffffff

const CL_DEVICE_TYPE = 0x1000

const CL_DEVICE_VENDOR_ID = 0x1001

const CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002

const CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = 0x1003

const CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004

const CL_DEVICE_MAX_WORK_ITEM_SIZES = 0x1005

const CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR = 0x1006

const CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT = 0x1007

const CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT = 0x1008

const CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG = 0x1009

const CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT = 0x100a

const CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE = 0x100b

const CL_DEVICE_MAX_CLOCK_FREQUENCY = 0x100c

const CL_DEVICE_ADDRESS_BITS = 0x100d

const CL_DEVICE_MAX_READ_IMAGE_ARGS = 0x100e

const CL_DEVICE_MAX_WRITE_IMAGE_ARGS = 0x100f

const CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010

const CL_DEVICE_IMAGE2D_MAX_WIDTH = 0x1011

const CL_DEVICE_IMAGE2D_MAX_HEIGHT = 0x1012

const CL_DEVICE_IMAGE3D_MAX_WIDTH = 0x1013

const CL_DEVICE_IMAGE3D_MAX_HEIGHT = 0x1014

const CL_DEVICE_IMAGE3D_MAX_DEPTH = 0x1015

const CL_DEVICE_IMAGE_SUPPORT = 0x1016

const CL_DEVICE_MAX_PARAMETER_SIZE = 0x1017

const CL_DEVICE_MAX_SAMPLERS = 0x1018

const CL_DEVICE_MEM_BASE_ADDR_ALIGN = 0x1019

const CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE = 0x101a

const CL_DEVICE_SINGLE_FP_CONFIG = 0x101b

const CL_DEVICE_GLOBAL_MEM_CACHE_TYPE = 0x101c

const CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE = 0x101d

const CL_DEVICE_GLOBAL_MEM_CACHE_SIZE = 0x101e

const CL_DEVICE_GLOBAL_MEM_SIZE = 0x101f

const CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = 0x1020

const CL_DEVICE_MAX_CONSTANT_ARGS = 0x1021

const CL_DEVICE_LOCAL_MEM_TYPE = 0x1022

const CL_DEVICE_LOCAL_MEM_SIZE = 0x1023

const CL_DEVICE_ERROR_CORRECTION_SUPPORT = 0x1024

const CL_DEVICE_PROFILING_TIMER_RESOLUTION = 0x1025

const CL_DEVICE_ENDIAN_LITTLE = 0x1026

const CL_DEVICE_AVAILABLE = 0x1027

const CL_DEVICE_COMPILER_AVAILABLE = 0x1028

const CL_DEVICE_EXECUTION_CAPABILITIES = 0x1029

const CL_DEVICE_QUEUE_PROPERTIES = 0x102a

const CL_DEVICE_QUEUE_ON_HOST_PROPERTIES = 0x102a

const CL_DEVICE_NAME = 0x102b

const CL_DEVICE_VENDOR = 0x102c

const CL_DRIVER_VERSION = 0x102d

const CL_DEVICE_PROFILE = 0x102e

const CL_DEVICE_VERSION = 0x102f

const CL_DEVICE_EXTENSIONS = 0x1030

const CL_DEVICE_PLATFORM = 0x1031

const CL_DEVICE_DOUBLE_FP_CONFIG = 0x1032

const CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF = 0x1034

const CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035

const CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR = 0x1036

const CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT = 0x1037

const CL_DEVICE_NATIVE_VECTOR_WIDTH_INT = 0x1038

const CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG = 0x1039

const CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT = 0x103a

const CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE = 0x103b

const CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF = 0x103c

const CL_DEVICE_OPENCL_C_VERSION = 0x103d

const CL_DEVICE_LINKER_AVAILABLE = 0x103e

const CL_DEVICE_BUILT_IN_KERNELS = 0x103f

const CL_DEVICE_IMAGE_MAX_BUFFER_SIZE = 0x1040

const CL_DEVICE_IMAGE_MAX_ARRAY_SIZE = 0x1041

const CL_DEVICE_PARENT_DEVICE = 0x1042

const CL_DEVICE_PARTITION_MAX_SUB_DEVICES = 0x1043

const CL_DEVICE_PARTITION_PROPERTIES = 0x1044

const CL_DEVICE_PARTITION_AFFINITY_DOMAIN = 0x1045

const CL_DEVICE_PARTITION_TYPE = 0x1046

const CL_DEVICE_REFERENCE_COUNT = 0x1047

const CL_DEVICE_PREFERRED_INTEROP_USER_SYNC = 0x1048

const CL_DEVICE_PRINTF_BUFFER_SIZE = 0x1049

const CL_DEVICE_IMAGE_PITCH_ALIGNMENT = 0x104a

const CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT = 0x104b

const CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS = 0x104c

const CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE = 0x104d

const CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES = 0x104e

const CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE = 0x104f

const CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE = 0x1050

const CL_DEVICE_MAX_ON_DEVICE_QUEUES = 0x1051

const CL_DEVICE_MAX_ON_DEVICE_EVENTS = 0x1052

const CL_DEVICE_SVM_CAPABILITIES = 0x1053

const CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE = 0x1054

const CL_DEVICE_MAX_PIPE_ARGS = 0x1055

const CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS = 0x1056

const CL_DEVICE_PIPE_MAX_PACKET_SIZE = 0x1057

const CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT = 0x1058

const CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT = 0x1059

const CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT = 0x105a

const CL_DEVICE_IL_VERSION = 0x105b

const CL_DEVICE_MAX_NUM_SUB_GROUPS = 0x105c

const CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS = 0x105d

const CL_DEVICE_NUMERIC_VERSION = 0x105e

const CL_DEVICE_EXTENSIONS_WITH_VERSION = 0x1060

const CL_DEVICE_ILS_WITH_VERSION = 0x1061

const CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION = 0x1062

const CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES = 0x1063

const CL_DEVICE_ATOMIC_FENCE_CAPABILITIES = 0x1064

const CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT = 0x1065

const CL_DEVICE_OPENCL_C_ALL_VERSIONS = 0x1066

const CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x1067

const CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT = 0x1068

const CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT = 0x1069

const CL_DEVICE_OPENCL_C_FEATURES = 0x106f

const CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES = 0x1070

const CL_DEVICE_PIPE_SUPPORT = 0x1071

const CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED = 0x1072

const CL_PROGRAM_REFERENCE_COUNT = 0x1160

const CL_PROGRAM_CONTEXT = 0x1161

const CL_PROGRAM_NUM_DEVICES = 0x1162

const CL_PROGRAM_DEVICES = 0x1163

const CL_PROGRAM_SOURCE = 0x1164

const CL_PROGRAM_BINARY_SIZES = 0x1165

const CL_PROGRAM_BINARIES = 0x1166

const CL_PROGRAM_NUM_KERNELS = 0x1167

const CL_PROGRAM_KERNEL_NAMES = 0x1168

const CL_PROGRAM_IL = 0x1169

const CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT = 0x116a

const CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT = 0x116b

const CL_PROGRAM_BUILD_STATUS = 0x1181

const CL_PROGRAM_BUILD_OPTIONS = 0x1182

const CL_PROGRAM_BUILD_LOG = 0x1183

const CL_PROGRAM_BINARY_TYPE = 0x1184

const CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE = 0x1185

const CL_PROGRAM_BINARY_TYPE_NONE = 0x00

const CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT = 0x01

const CL_PROGRAM_BINARY_TYPE_LIBRARY = 0x02

const CL_PROGRAM_BINARY_TYPE_EXECUTABLE = 0x04

const CL_BUILD_SUCCESS = 0

const CL_BUILD_NONE = -1

const CL_BUILD_ERROR = -2

const CL_BUILD_IN_PROGRESS = -3

const CL_KERNEL_WORK_GROUP_SIZE = 0x11b0

const CL_KERNEL_COMPILE_WORK_GROUP_SIZE = 0x11b1

const CL_KERNEL_LOCAL_MEM_SIZE = 0x11b2

const CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x11b3

const CL_KERNEL_PRIVATE_MEM_SIZE = 0x11b4

const CL_KERNEL_GLOBAL_WORK_SIZE = 0x11b5

const CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE = 0x2033

const CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE = 0x2034

const CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT = 0x11b8

const CL_KERNEL_MAX_NUM_SUB_GROUPS = 0x11b9

const CL_KERNEL_COMPILE_NUM_SUB_GROUPS = 0x11ba

const CL_KERNEL_EXEC_INFO_SVM_PTRS = 0x11b6

const CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM = 0x11b7

struct CLError <: Exception
    code::Cint
end

@noinline function throw_api_error(res)
    throw(CLError(res))
end

function check(f)
    res = f()

    if res != CL_SUCCESS
        throw_api_error(res)
    end

    return
end

const intptr_t = if sizeof(Ptr{Cvoid}) == 8
    Int64
else
    Int32
end

const cl_int = Int32

const cl_uint = UInt32

const cl_ulong = UInt64

mutable struct _cl_platform_id end

mutable struct _cl_device_id end

mutable struct _cl_context end

mutable struct _cl_command_queue end

mutable struct _cl_mem end

mutable struct _cl_program end

mutable struct _cl_kernel end

mutable struct _cl_event end

const cl_platform_id = Ptr{_cl_platform_id}

const cl_device_id = Ptr{_cl_device_id}

const cl_context = Ptr{_cl_context}

const cl_command_queue = Ptr{_cl_command_queue}

const cl_mem = Ptr{_cl_mem}

const cl_program = Ptr{_cl_program}

const cl_kernel = Ptr{_cl_kernel}

const cl_event = Ptr{_cl_event}

const cl_bitfield = cl_ulong

const cl_device_type = cl_bitfield

const cl_platform_info = cl_uint

const cl_device_info = cl_uint

const cl_context_properties = intptr_t

const cl_context_info = cl_uint

const cl_build_status = cl_int

const cl_program_info = cl_uint

const cl_program_build_info = cl_uint

const cl_kernel_info = cl_uint

const cl_kernel_arg_info = cl_uint

const cl_kernel_arg_address_qualifier = cl_uint

const cl_kernel_arg_access_qualifier = cl_uint

const cl_kernel_arg_type_qualifier = cl_bitfield

const cl_kernel_work_group_info = cl_uint

const cl_kernel_sub_group_info = cl_uint

const cl_device_svm_capabilities = cl_bitfield

const cl_command_queue_properties = cl_bitfield

const cl_event_info = cl_uint

@checked function clGetPlatformIDs(num_entries, platforms, num_platforms)
    @ccall libopencl.clGetPlatformIDs(
        num_entries::cl_uint, platforms::Ptr{cl_platform_id},
        num_platforms::Ptr{cl_uint}
    )::cl_int
end

@checked function clGetPlatformInfo(
        platform, param_name, param_value_size, param_value,
        param_value_size_ret
    )
    @ccall libopencl.clGetPlatformInfo(
        platform::cl_platform_id,
        param_name::cl_platform_info,
        param_value_size::Csize_t, param_value::Ptr{Cvoid},
        param_value_size_ret::Ptr{Csize_t}
    )::cl_int
end

@checked function clGetDeviceIDs(platform, device_type, num_entries, devices, num_devices)
    @ccall libopencl.clGetDeviceIDs(
        platform::cl_platform_id, device_type::cl_device_type,
        num_entries::cl_uint, devices::Ptr{cl_device_id},
        num_devices::Ptr{cl_uint}
    )::cl_int
end

@checked function clGetDeviceInfo(
        device, param_name, param_value_size, param_value,
        param_value_size_ret
    )
    @ccall libopencl.clGetDeviceInfo(
        device::cl_device_id, param_name::cl_device_info,
        param_value_size::Csize_t, param_value::Ptr{Cvoid},
        param_value_size_ret::Ptr{Csize_t}
    )::cl_int
end

function clCreateContext(
        properties, num_devices, devices, pfn_notify, user_data,
        errcode_ret
    )
    return @ccall libopencl.clCreateContext(
        properties::Ptr{cl_context_properties},
        num_devices::cl_uint, devices::Ptr{cl_device_id},
        pfn_notify::Ptr{Cvoid}, user_data::Ptr{Cvoid},
        errcode_ret::Ptr{cl_int}
    )::cl_context
end

@checked function clReleaseContext(context)
    @ccall libopencl.clReleaseContext(context::cl_context)::cl_int
end

function clCreateProgramWithIL(context, il, length, errcode_ret)
    return @ccall libopencl.clCreateProgramWithIL(
        context::cl_context, il::Ptr{Cvoid},
        length::Csize_t,
        errcode_ret::Ptr{cl_int}
    )::cl_program
end

@checked function clReleaseProgram(program)
    @ccall libopencl.clReleaseProgram(program::cl_program)::cl_int
end

@checked function clBuildProgram(
        program, num_devices, device_list, options, pfn_notify,
        user_data
    )
    @ccall libopencl.clBuildProgram(
        program::cl_program, num_devices::cl_uint,
        device_list::Ptr{cl_device_id}, options::Ptr{Cchar},
        pfn_notify::Ptr{Cvoid}, user_data::Ptr{Cvoid}
    )::cl_int
end

@checked function clGetProgramInfo(
        program, param_name, param_value_size, param_value,
        param_value_size_ret
    )
    @ccall libopencl.clGetProgramInfo(
        program::cl_program, param_name::cl_program_info,
        param_value_size::Csize_t, param_value::Ptr{Cvoid},
        param_value_size_ret::Ptr{Csize_t}
    )::cl_int
end

@checked function clGetProgramBuildInfo(
        program, device, param_name, param_value_size,
        param_value, param_value_size_ret
    )
    @ccall libopencl.clGetProgramBuildInfo(
        program::cl_program, device::cl_device_id,
        param_name::cl_program_build_info,
        param_value_size::Csize_t,
        param_value::Ptr{Cvoid},
        param_value_size_ret::Ptr{Csize_t}
    )::cl_int
end

function clCreateKernel(program, kernel_name, errcode_ret)
    return @ccall libopencl.clCreateKernel(
        program::cl_program, kernel_name::Ptr{Cchar},
        errcode_ret::Ptr{cl_int}
    )::cl_kernel
end

@checked function clReleaseKernel(kernel)
    @ccall libopencl.clReleaseKernel(kernel::cl_kernel)::cl_int
end

@checked function clSetKernelArg(kernel, arg_index, arg_size, arg_value)
    @ccall libopencl.clSetKernelArg(
        kernel::cl_kernel, arg_index::cl_uint,
        arg_size::Csize_t, arg_value::Ptr{Cvoid}
    )::cl_int
end

@checked function clSetKernelArgSVMPointer(kernel, arg_index, arg_value)
    @ccall libopencl.clSetKernelArgSVMPointer(
        kernel::cl_kernel, arg_index::cl_uint,
        arg_value::Ptr{Cvoid}
    )::cl_int
end

@checked function clGetKernelWorkGroupInfo(
        kernel, device, param_name, param_value_size,
        param_value, param_value_size_ret
    )
    @ccall libopencl.clGetKernelWorkGroupInfo(
        kernel::cl_kernel, device::cl_device_id,
        param_name::cl_kernel_work_group_info,
        param_value_size::Csize_t,
        param_value::Ptr{Cvoid},
        param_value_size_ret::Ptr{Csize_t}
    )::cl_int
end

@checked function clEnqueueNDRangeKernel(
        command_queue, kernel, work_dim,
        global_work_offset, global_work_size,
        local_work_size, num_events_in_wait_list,
        event_wait_list, event
    )
    @ccall libopencl.clEnqueueNDRangeKernel(
        command_queue::cl_command_queue,
        kernel::cl_kernel, work_dim::cl_uint,
        global_work_offset::Ptr{Csize_t},
        global_work_size::Ptr{Csize_t},
        local_work_size::Ptr{Csize_t},
        num_events_in_wait_list::cl_uint,
        event_wait_list::Ptr{cl_event},
        event::Ptr{cl_event}
    )::cl_int
end

function clCreateCommandQueue(context, device, properties, errcode_ret)
    return @ccall libopencl.clCreateCommandQueue(
        context::cl_context, device::cl_device_id,
        properties::cl_command_queue_properties,
        errcode_ret::Ptr{cl_int}
    )::cl_command_queue
end

@checked function clReleaseCommandQueue(command_queue)
    @ccall libopencl.clReleaseCommandQueue(command_queue::cl_command_queue)::cl_int
end

@checked function clFinish(command_queue)
    @ccall libopencl.clFinish(command_queue::cl_command_queue)::cl_int
end

@checked function clWaitForEvents(num_events, event_list)
    @ccall libopencl.clWaitForEvents(num_events::cl_uint, event_list::Ptr{cl_event})::cl_int
end

@checked function clGetEventInfo(
        event, param_name, param_value_size, param_value,
        param_value_size_ret
    )
    @ccall libopencl.clGetEventInfo(
        event::cl_event, param_name::cl_event_info,
        param_value_size::Csize_t, param_value::Ptr{Cvoid},
        param_value_size_ret::Ptr{Csize_t}
    )::cl_int
end

@checked function clReleaseEvent(event)
    @ccall libopencl.clReleaseEvent(event::cl_event)::cl_int
end

# Init

# lazy initialization
const initialized = Ref{Bool}(false)
@noinline function initialize()
    initialized[] = true

    # @static if Sys.iswindows()
    #     if is_high_integrity_level()
    #         @warn """Running at high integrity level, preventing OpenCL.jl from loading drivers from JLLs.

    #         Only system drivers will be available. To enable JLL drivers, do not run Julia as an administrator."""
    #     end
    # end

    ocd_filenames = join(OpenCL_jll.drivers, ':')
    if haskey(ENV, "OCL_ICD_FILENAMES")
        ocd_filenames *= ":" * ENV["OCL_ICD_FILENAMES"]
    end

    return withenv("OCL_ICD_FILENAMES" => ocd_filenames) do
        num_platforms = Ref{Cuint}()
        @ccall libopencl.clGetPlatformIDs(
            0::cl_uint, C_NULL::Ptr{cl_platform_id},
            num_platforms::Ptr{cl_uint}
        )::cl_int

        if num_platforms[] == 0 && isempty(OpenCL_jll.drivers)
            @error """No OpenCL drivers available, either system-wide or provided by a JLL.

            Please install a system-wide OpenCL driver, or load one together with OpenCL.jl,
            e.g., by doing `using OpenCL, pocl_jll`."""
        end
    end
end

# Julia API

struct Platform
    id::cl_platform_id
end

Base.unsafe_convert(::Type{cl_platform_id}, p::Platform) = p.id

function platforms()
    nplatforms = Ref{Cuint}()
    res = unchecked_clGetPlatformIDs(0, C_NULL, nplatforms)
    if res == CL_PLATFORM_NOT_FOUND_KHR || nplatforms[] == 0
        return Platform[]
    elseif res != CL_SUCCESS
        throw(CLError(res))
    end
    cl_platform_ids = Vector{cl_platform_id}(undef, nplatforms[])
    clGetPlatformIDs(nplatforms[], cl_platform_ids, C_NULL)
    return [Platform(id) for id in cl_platform_ids]
end


function Base.getproperty(p::Platform, s::Symbol)
    # simple string properties
    version_re = r"OpenCL (?<major>\d+)\.(?<minor>\d+)(?<vendor>.+)"
    @inline function get_string(prop)
        sz = Ref{Csize_t}()
        clGetPlatformInfo(p, prop, 0, C_NULL, sz)
        chars = Vector{Cchar}(undef, sz[])
        clGetPlatformInfo(p, prop, sz[], chars, C_NULL)
        return GC.@preserve chars unsafe_string(pointer(chars))
    end
    if s === :profile
        return get_string(CL_PLATFORM_PROFILE)
    elseif s === :version
        str = get_string(CL_PLATFORM_VERSION)
        m = match(version_re, str)
        if m === nothing
            error("Could not parse OpenCL version string: $str")
        end
        return strip(m["vendor"])
    elseif s === :opencl_version
        str = get_string(CL_PLATFORM_VERSION)
        m = match(version_re, str)
        if m === nothing
            error("Could not parse OpenCL version string: $str")
        end
        return VersionNumber(parse(Int, m["major"]), parse(Int, m["minor"]))
    elseif s === :name
        return get_string(CL_PLATFORM_NAME)
    elseif s === :vendor
        return get_string(CL_PLATFORM_VENDOR)
    end

    if s == :extensions
        size = Ref{Csize_t}()
        clGetPlatformInfo(p, CL_PLATFORM_EXTENSIONS, 0, C_NULL, size)
        result = Vector{Cchar}(undef, size[])
        clGetPlatformInfo(p, CL_PLATFORM_EXTENSIONS, size[], result, C_NULL)
        return GC.@preserve result split(unsafe_string(pointer(result)))
    end
    return getfield(p, s)
end

struct Device
    id::cl_device_id
end

Base.unsafe_convert(::Type{cl_device_id}, d::Device) = d.id

function devices(p::Platform, dtype)
    ndevices = Ref{Cuint}()
    ret = unchecked_clGetDeviceIDs(p, dtype, 0, C_NULL, ndevices)
    if ret == CL_DEVICE_NOT_FOUND || ndevices[] == 0
        return Device[]
    elseif ret != CL_SUCCESS
        throw(CLError(ret))
    end
    result = Vector{cl_device_id}(undef, ndevices[])
    clGetDeviceIDs(p, dtype, ndevices[], result, C_NULL)
    return Device[Device(id) for id in result]
end

function default_device(p::Platform)
    devs = devices(p, CL_DEVICE_TYPE_DEFAULT)
    isempty(devs) && return nothing
    # XXX: clGetDeviceIDs documents CL_DEVICE_TYPE_DEFAULT should only return one device,
    #      but it's been observed to return multiple devices on some platforms...
    return first(devs)
end

devices(p::Platform) = devices(p, CL_DEVICE_TYPE_ALL)

@inline function Base.getproperty(d::Device, s::Symbol)
    # simple string properties
    version_re = r"OpenCL (?<major>\d+)\.(?<minor>\d+)(?<vendor>.+)"
    @inline function get_string(prop)
        sz = Ref{Csize_t}()
        clGetDeviceInfo(d, prop, 0, C_NULL, sz)
        chars = Vector{Cchar}(undef, sz[])
        clGetDeviceInfo(d, prop, sz[], chars, C_NULL)
        return GC.@preserve chars unsafe_string(pointer(chars))
    end
    if s === :profile
        return get_string(CL_DEVICE_PROFILE)
    elseif s === :version
        str = get_string(CL_DEVICE_VERSION)
        m = match(version_re, str)
        if m === nothing
            error("Could not parse OpenCL version string: $str")
        end
        return strip(m["vendor"])
    elseif s === :opencl_version
        str = get_string(CL_DEVICE_VERSION)
        m = match(version_re, str)
        if m === nothing
            error("Could not parse OpenCL version string: $str")
        end
        return VersionNumber(parse(Int, m["major"]), parse(Int, m["minor"]))
    elseif s === :driver_version
        return get_string(CL_DRIVER_VERSION)
    elseif s === :name
        return get_string(CL_DEVICE_NAME)
    end

    # scalar values
    @inline function get_scalar(prop, typ)
        scalar = Ref{typ}()
        clGetDeviceInfo(d, prop, sizeof(typ), scalar, C_NULL)
        return Int(scalar[])
    end
    if s === :vendor_id
        return get_scalar(CL_DEVICE_VENDOR_ID, cl_uint)
    elseif s === :max_compute_units
        return get_scalar(CL_DEVICE_MAX_COMPUTE_UNITS, cl_uint)
    elseif s === :max_work_item_dims
        return get_scalar(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, cl_uint)
    elseif s === :max_clock_frequency
        return get_scalar(CL_DEVICE_MAX_CLOCK_FREQUENCY, cl_uint)
    elseif s === :address_bits
        return get_scalar(CL_DEVICE_ADDRESS_BITS, cl_uint)
    elseif s === :max_read_image_args
        return get_scalar(CL_DEVICE_MAX_READ_IMAGE_ARGS, cl_uint)
    elseif s === :max_write_image_args
        return get_scalar(CL_DEVICE_MAX_WRITE_IMAGE_ARGS, cl_uint)
    elseif s === :global_mem_size
        return get_scalar(CL_DEVICE_GLOBAL_MEM_SIZE, cl_ulong)
    elseif s === :max_mem_alloc_size
        return get_scalar(CL_DEVICE_MAX_MEM_ALLOC_SIZE, cl_ulong)
    elseif s === :max_const_buffer_size
        return get_scalar(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, cl_ulong)
    elseif s === :local_mem_size
        return get_scalar(CL_DEVICE_LOCAL_MEM_SIZE, cl_ulong)
    elseif s === :max_work_group_size
        return get_scalar(CL_DEVICE_MAX_WORK_GROUP_SIZE, Csize_t)
    elseif s === :max_parameter_size
        return get_scalar(CL_DEVICE_MAX_PARAMETER_SIZE, Csize_t)
    elseif s === :profiling_timer_resolution
        return get_scalar(CL_DEVICE_PROFILING_TIMER_RESOLUTION, Csize_t)
    end

    # boolean properties
    @inline function get_bool(prop)
        bool = Ref{cl_bool}()
        clGetDeviceInfo(d, prop, sizeof(cl_bool), bool, C_NULL)
        return bool[] == CL_TRUE
    end
    if s === :has_image_support
        return get_bool(CL_DEVICE_IMAGE_SUPPORT)
    elseif s === :has_local_mem
        return get_bool(CL_DEVICE_LOCAL_MEM_TYPE)
    elseif s === :host_unified_memory
        return get_bool(CL_DEVICE_HOST_UNIFIED_MEMORY)
    elseif s === :available
        return get_bool(CL_DEVICE_AVAILABLE)
    elseif s === :compiler_available
        return get_bool(CL_DEVICE_COMPILER_AVAILABLE)
    end

    if s == :extensions
        size = Ref{Csize_t}()
        clGetDeviceInfo(d, CL_DEVICE_EXTENSIONS, 0, C_NULL, size)
        result = Vector{Cchar}(undef, size[])
        clGetDeviceInfo(d, CL_DEVICE_EXTENSIONS, size[], result, C_NULL)
        bs = GC.@preserve result unsafe_string(pointer(result))
        return String[string(s) for s in split(bs)]
    end

    if s == :platform
        result = Ref{cl_platform_id}()
        clGetDeviceInfo(d, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), result, C_NULL)
        return Platform(result[])
    end

    if s == :device_type
        result = Ref{cl_device_type}()
        clGetDeviceInfo(d, CL_DEVICE_TYPE, sizeof(cl_device_type), result, C_NULL)
        result = result[]
        if result == CL_DEVICE_TYPE_GPU
            return :gpu
        elseif result == CL_DEVICE_TYPE_CPU
            return :cpu
        elseif result == CL_DEVICE_TYPE_ACCELERATOR
            return :accelerator
        elseif result == CL_DEVICE_TYPE_CUSTOM
            return :custom
        else
            return :unknown
        end
    end

    if s == :max_work_item_size
        result = Vector{Csize_t}(undef, d.max_work_item_dims)
        clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(result), result, C_NULL)
        return tuple([Int(r) for r in result]...)
    end

    if s == :max_image2d_shape
        width = Ref{Csize_t}()
        height = Ref{Csize_t}()
        clGetDeviceInfo(d, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(Csize_t), width, C_NULL)
        clGetDeviceInfo(d, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(Csize_t), height, C_NULL)
        return (width[], height[])
    end

    if s == :max_image3d_shape
        width = Ref{Csize_t}()
        height = Ref{Csize_t}()
        depth = Ref{Csize_t}()
        clGetDeviceInfo(d, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(Csize_t), width, C_NULL)
        clGetDeviceInfo(d, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(Csize_t), height, C_NULL)
        clGetDeviceInfo(d, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(Csize_t), depth, C_NULL)
        return (width[], height[], depth[])
    end

    return getfield(d, s)
end

mutable struct Context
    const id::cl_context

    function Context(ctx_id::cl_context)
        ctx = new(ctx_id)
        finalizer(clReleaseContext, ctx)
        return ctx
    end
end

Base.unsafe_convert(::Type{cl_context}, ctx::Context) = ctx.id

function Context(device::Device)
    device_id = Ref(device.id)

    err_code = Ref{Cint}()
    ctx_id = clCreateContext(
        C_NULL, 1, device_id, C_NULL, C_NULL, err_code
    )
    if err_code[] != CL_SUCCESS
        throw(CLError(err_code[]))
    end
    return Context(ctx_id)
end

mutable struct Program
    const id::cl_program

    function Program(program_id::cl_program)
        p = new(program_id)
        finalizer(clReleaseProgram, p)
        return p
    end
end

Base.unsafe_convert(::Type{cl_program}, p::Program) = p.id

function Program(il, ctx)
    err_code = Ref{Cint}()
    program_id = clCreateProgramWithIL(ctx, il, length(il), err_code)
    if err_code[] != CL_SUCCESS
        throw(CLError(err_code[]))
    end
    return Program(program_id)
end

#TODO: build callback...
function build!(p::Program; options = "")
    opts = String(options)
    ndevices = 0
    device_ids = C_NULL
    try
        clBuildProgram(p, cl_uint(ndevices), device_ids, opts, C_NULL, C_NULL)
    catch err
        isa(err, CLError) || throw(err)

        for (dev, status) in p.build_status
            if status == CL_BUILD_ERROR
                io = IOBuffer()
                println(io, "Failed to compile program")
                if p.source !== nothing
                    println(io)
                    println(io, "Source code:")
                    for (i, line) in enumerate(split(p.source, "\n"))
                        println(io, @sprintf("%s%-2d: %s", " ", i, line))
                    end
                end
                println(io)
                println(io, "Build log:")
                println(io, strip(p.build_log[dev]))
                error(String(take!(io)))
            end
        end
    end
    return p
end

function Base.getproperty(p::Program, s::Symbol)
    if s == :reference_count
        count = Ref{Cuint}()
        clGetProgramInfo(p, CL_PROGRAM_REFERENCE_COUNT, sizeof(Cuint), count, C_NULL)
        return Int(count[])
    elseif s == :num_devices
        count = Ref{Cuint}()
        clGetProgramInfo(p, CL_PROGRAM_NUM_DEVICES, sizeof(Cuint), count, C_NULL)
        return Int(count[])
    elseif s == :devices
        device_ids = Vector{cl_device_id}(undef, p.num_devices)
        clGetProgramInfo(p, CL_PROGRAM_DEVICES, sizeof(device_ids), device_ids, C_NULL)
        return [Device(id) for id in device_ids]
    elseif s == :source
        src_len = Ref{Csize_t}()
        clGetProgramInfo(p, CL_PROGRAM_SOURCE, 0, C_NULL, src_len)
        src_len[] <= 1 && return nothing
        src = Vector{Cchar}(undef, src_len[])
        clGetProgramInfo(p, CL_PROGRAM_SOURCE, src_len[], src, C_NULL)
        return GC.@preserve src unsafe_string(pointer(src))
    elseif s == :binary_sizes
        sizes = Vector{Csize_t}(undef, p.num_devices)
        clGetProgramInfo(p, CL_PROGRAM_BINARY_SIZES, sizeof(sizes), sizes, C_NULL)
        return sizes
    elseif s == :binaries
        sizes = p.binary_sizes

        bins = Vector{Ptr{UInt8}}(undef, length(sizes))
        # keep a reference to the underlying binary arrays
        # as storing the pointer to the array hides the additional
        # reference from julia's garbage collector
        bin_arrays = Any[]
        for (i, s) in enumerate(sizes)
            if s > 0
                bin = Vector{UInt8}(undef, s)
                bins[i] = pointer(bin)
                push!(bin_arrays, bin)
            else
                bins[i] = Base.unsafe_convert(Ptr{UInt8}, C_NULL)
            end
        end
        clGetProgramInfo(p, CL_PROGRAM_BINARIES, sizeof(bins), bins, C_NULL)

        binary_dict = Dict{Device, Array{UInt8}}()
        bidx = 1
        for (i, d) in enumerate(p.devices)
            if sizes[i] > 0
                binary_dict[d] = bin_arrays[bidx]
                bidx += 1
            end
        end
        return binary_dict
    elseif s == :context
        ctx = Ref{cl_context}()
        clGetProgramInfo(p, CL_PROGRAM_CONTEXT, sizeof(cl_context), ctx, C_NULL)
        return Context(ctx[], retain = true)
    elseif s == :build_status
        status_dict = Dict{Device, cl_build_status}()
        for device in p.devices
            status = Ref{cl_build_status}()
            clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), status, C_NULL)
            status_dict[device] = status[]
        end
        return status_dict
    elseif s == :build_log
        log_dict = Dict{Device, String}()
        for device in p.devices
            size = Ref{Csize_t}()
            clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_LOG, 0, C_NULL, size)
            log = Vector{Cchar}(undef, size[])
            clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_LOG, size[], log, C_NULL)
            log_dict[device] = GC.@preserve log unsafe_string(pointer(log))
        end
        return log_dict
    else
        return getfield(p, s)
    end
end

mutable struct Kernel
    const id::cl_kernel

    function Kernel(k::cl_kernel)
        kernel = new(k)
        finalizer(clReleaseKernel, kernel)
        return kernel
    end
end

Base.unsafe_convert(::Type{cl_kernel}, k::Kernel) = k.id

function Kernel(p::Program, kernel_name::String)
    for (dev, status) in p.build_status
        if status != CL_BUILD_SUCCESS
            msg = "OpenCL.Program has to be built before Kernel constructor invoked"
            throw(ArgumentError(msg))
        end
    end
    err_code = Ref{Cint}()
    kernel_id = clCreateKernel(p, kernel_name, err_code)
    if err_code[] != CL_SUCCESS
        throw(CLError(err_code[]))
    end
    return Kernel(kernel_id)
end

struct LocalMem{T}
    nbytes::Csize_t
end

function LocalMem(::Type{T}, len::Integer) where {T}
    @assert len > 0
    nbytes = sizeof(T) * len
    return LocalMem{T}(convert(Csize_t, nbytes))
end

Base.ndims(l::LocalMem) = 1
Base.eltype(l::LocalMem{T}) where {T} = T
Base.sizeof(l::LocalMem{T}) where {T} = l.nbytes
Base.length(l::LocalMem{T}) where {T} = Int(l.nbytes ÷ sizeof(T))

# preserve the LocalMem; it will be handled by set_arg!
# XXX: do we want set_arg!(C_NULL::Ptr) to just call clSetKernelArg?
Base.unsafe_convert(::Type{Ptr{T}}, l::LocalMem{T}) where {T} = l

function set_arg!(k::Kernel, idx::Integer, arg::Nothing)
    @assert idx > 0
    clSetKernelArg(k, cl_uint(idx - 1), sizeof(cl_mem), C_NULL)
    return k
end

# SVMBuffers
## when passing using `cl.call`
# function set_arg!(k::Kernel, idx::Integer, arg::SVMBuffer)
#     clSetKernelArgSVMPointer(k, cl_uint(idx-1), arg.ptr)
#     return k
# end
## when passing with `clcall`, which has pre-converted the buffer
function set_arg!(k::Kernel, idx::Integer, arg::Union{Ptr, Core.LLVMPtr})
    arg = reinterpret(Ptr{Cvoid}, arg)
    if arg != C_NULL
        # XXX: this assumes that the receiving argument is pointer-typed, which is not the
        #      case with Julia's `Ptr` ABI. Instead, one should reinterpret the pointer as a
        #      `Core.LLVMPtr`, which _is_ pointer-valued. We retain this handling for `Ptr`
        #      for users passing pointers to OpenCL C, and because `Ptr` is pointer-valued
        #      starting with Julia 1.12.
        clSetKernelArgSVMPointer(k, cl_uint(idx - 1), arg)
    end
    return k
end

# regular buffers
# function set_arg!(k::Kernel, idx::Integer, arg::AbstractMemory)
#     arg_boxed = Ref(arg.id)
#     clSetKernelArg(k, cl_uint(idx-1), sizeof(cl_mem), arg_boxed)
#     return k
# end

function set_arg!(k::Kernel, idx::Integer, arg::LocalMem)
    clSetKernelArg(k, cl_uint(idx - 1), arg.nbytes, C_NULL)
    return k
end

function set_arg!(k::Kernel, idx::Integer, arg::T) where {T}
    ref = Ref(arg)
    tsize = sizeof(ref)
    err = unchecked_clSetKernelArg(k, cl_uint(idx - 1), tsize, ref)
    if err == CL_INVALID_ARG_SIZE
        error(
            """Mismatch between Julia and OpenCL type for kernel argument $idx.

            Possible reasons:
            - OpenCL does not support empty types.
            - Vectors of length 3 (e.g., `float3`) are packed as 4-element vectors;
              consider padding your tuples.
            - The alignment of fields in your struct may not match the OpenCL layout.
              Make sure your Julia definition matches the OpenCL layout, e.g., by
              using `__attribute__((packed))` in your OpenCL struct definition."""
        )
    elseif err != CL_SUCCESS
        throw(CLError(err))
    end
    return k
end

function set_args!(k::Kernel, args...)
    for (i, a) in enumerate(args)
        set_arg!(k, i, a)
    end
    return
end

function enqueue_kernel(
        k::Kernel, global_work_size, local_work_size = nothing;
        global_work_offset = nothing
    )
    max_work_dim = device().max_work_item_dims
    work_dim = length(global_work_size)
    if work_dim > max_work_dim
        throw(ArgumentError("global_work_size has max dim of $max_work_dim"))
    end
    gsize = Vector{Csize_t}(undef, work_dim)
    for (i, s) in enumerate(global_work_size)
        gsize[i] = s
    end

    goffset = C_NULL
    if global_work_offset !== nothing
        if length(global_work_offset) > max_work_dim
            throw(ArgumentError("global_work_offset has max dim of $max_work_dim"))
        end
        if length(global_work_offset) != work_dim
            throw(ArgumentError("global_work_size and global_work_offset have differing dims"))
        end
        goffset = Vector{Csize_t}(undef, work_dim)
        for (i, o) in enumerate(global_work_offset)
            goffset[i] = o
        end
    else
        # null global offset means (0, 0, 0)
    end

    lsize = C_NULL
    if local_work_size !== nothing
        if length(local_work_size) > max_work_dim
            throw(ArgumentError("local_work_offset has max dim of $max_work_dim"))
        end
        if length(local_work_size) != work_dim
            throw(ArgumentError("global_work_size and local_work_size have differing dims"))
        end
        lsize = Vector{Csize_t}(undef, work_dim)
        for (i, s) in enumerate(local_work_size)
            lsize[i] = s
        end
    else
        # null local size means OpenCL decides
    end

    n_events = cl_uint(0)
    wait_event_ids = C_NULL
    ret_event = Ref{cl_event}()

    clEnqueueNDRangeKernel(
        queue(), k, cl_uint(work_dim), goffset, gsize, lsize,
        n_events, wait_event_ids, ret_event
    )
    return Event(ret_event[])
end

function call(
        k::Kernel, args...; global_size = (1,), local_size = nothing,
        global_work_offset = nothing,
        svm_pointers::Vector{Ptr{Cvoid}} = Ptr{Cvoid}[]
    )
    set_args!(k, args...)
    if !isempty(svm_pointers)
        clSetKernelExecInfo(
            k, CL_KERNEL_EXEC_INFO_SVM_PTRS,
            sizeof(svm_pointers), svm_pointers
        )
    end
    return enqueue_kernel(k, global_size, local_size; global_work_offset)
end

# convert the argument values to match the kernel's signature (specified by the user)
# (this mimics `lower-ccall` in julia-syntax.scm)
@inline @generated function convert_arguments(f::Function, ::Type{tt}, args...) where {tt}
    types = tt.parameters

    ex = quote end

    converted_args = Vector{Symbol}(undef, length(args))
    arg_ptrs = Vector{Symbol}(undef, length(args))
    for i in 1:length(args)
        converted_args[i] = gensym()
        arg_ptrs[i] = gensym()
        push!(ex.args, :($(converted_args[i]) = Base.cconvert($(types[i]), args[$i])))
        push!(ex.args, :($(arg_ptrs[i]) = Base.unsafe_convert($(types[i]), $(converted_args[i]))))
    end

    append!(
        ex.args, (
            quote
                GC.@preserve $(converted_args...) begin
                    f($(arg_ptrs...))
                end
            end
        ).args
    )

    return ex
end

clcall(f::F, types::Tuple, args::Vararg{Any, N}; kwargs...) where {N, F} =
    clcall(f, _to_tuple_type(types), args...; kwargs...)

function clcall(k::Kernel, types::Type{T}, args::Vararg{Any, N}; kwargs...) where {T, N}
    call_closure = function (converted_args::Vararg{Any, N})
        return call(k, converted_args...; kwargs...)
    end
    return convert_arguments(call_closure, types, args...)
end

struct KernelWorkGroupInfo
    kernel::Kernel
    device::Device
end
work_group_info(k::Kernel, d::Device) = KernelWorkGroupInfo(k, d)

function Base.getproperty(ki::KernelWorkGroupInfo, s::Symbol)
    k = getfield(ki, :kernel)
    d = getfield(ki, :device)

    function get(val, typ)
        result = Ref{typ}()
        clGetKernelWorkGroupInfo(k, d, val, sizeof(typ), result, C_NULL)
        return result[]
    end

    return if s == :size
        Int(get(CL_KERNEL_WORK_GROUP_SIZE, Csize_t))
    elseif s == :compile_size
        Int.(get(CL_KERNEL_COMPILE_WORK_GROUP_SIZE, NTuple{3, Csize_t}))
    elseif s == :local_mem_size
        Int(get(CL_KERNEL_LOCAL_MEM_SIZE, cl_ulong))
    elseif s == :private_mem_size
        Int(get(CL_KERNEL_PRIVATE_MEM_SIZE, cl_ulong))
    elseif s == :prefered_size_multiple
        Int(get(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, Csize_t))
    else
        getfield(ki, s)
    end
end

mutable struct CmdQueue
    const id::cl_command_queue

    function CmdQueue(q_id::cl_command_queue)
        q = new(q_id)
        finalizer(q) do _
            clReleaseCommandQueue(q)
        end
        return q
    end
end

Base.unsafe_convert(::Type{cl_command_queue}, q::CmdQueue) = q.id

function CmdQueue()
    flags = cl_command_queue_properties(0)
    err_code = Ref{Cint}()
    queue_id = clCreateCommandQueue(context(), device(), flags, err_code)
    if err_code[] != CL_SUCCESS
        if queue_id != C_NULL
            clReleaseCommandQueue(queue_id)
        end
        throw(CLError(err_code[]))
    end
    return CmdQueue(queue_id)
end

function finish(q::CmdQueue)
    clFinish(q)
    return q
end

struct Event
    id::cl_event
end
Base.unsafe_convert(::Type{cl_event}, e::Event) = e.id

const CL_EVENT_COMMAND_EXECUTION_STATUS = 0x11d3

function Base.getproperty(evt::Event, s::Symbol)
    # regular properties
    if s == :status
        st = Ref{Cint}()
        clGetEventInfo(evt, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(Cint), st, C_NULL)
        status = st[]
        return status
    else
        return getfield(evt, s)
    end
end

const CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST = -14

function Base.wait(evt::Event)
    evt_id = Ref(evt.id)
    err = unchecked_clWaitForEvents(cl_uint(1), evt_id)
    if err == CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST
        error("Kernel execution failed")
    elseif err != CL_SUCCESS
        throw(CLError(err))
    end
    return evt
end

end
