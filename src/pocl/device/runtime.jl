## exception mailbox

# Kernels report device-side exceptions into host memory whose address travels with the
# kernel state. pocl's CPU device shares the host address space, so the mailbox is an
# ordinary host allocation the kernel can reach by pointer. The host reads and resets it
# once every queue that could still write to it has completed; see `compiler/exceptions.jl`.

# Text capacities include the null terminator. Backtraces are recorded at debug level 2.
const EXCEPTION_NAME_LEN = 64
const EXCEPTION_REASON_LEN = 192
const EXCEPTION_FUNC_LEN = 128
const EXCEPTION_FILE_LEN = 128
const EXCEPTION_MAX_FRAMES = 16

struct ExceptionFrame_st
    func::NTuple{EXCEPTION_FUNC_LEN, UInt8}
    file::NTuple{EXCEPTION_FILE_LEN, UInt8}
    line::Int32

    ExceptionFrame_st() = new(
        ntuple(_ -> 0x00, EXCEPTION_FUNC_LEN),
        ntuple(_ -> 0x00, EXCEPTION_FILE_LEN), 0
    )
end

struct ExceptionInfo_st
    # whether an exception has been encountered (0 -> 1)
    status::Int32
    # claim state: 0 = unclaimed, 1 = claimed, 2 = owner published for re-entry
    output_lock::Int32
    # distinguish identical work-item coordinates in different kernel launches
    launch_id::UInt64
    # Keep positions at the builtins' 64-bit width and pad to four components. Narrowing
    # or merging adjacent three-component accesses can produce illegal six-wide SPIR-V
    # vectors. At level 2 these coordinates also identify the owner within a launch.
    work_item::NTuple{4, UInt64}
    work_group::NTuple{4, UInt64}
    # the exception type name and reason, as null-terminated text the host reads back
    name::NTuple{EXCEPTION_NAME_LEN, UInt8}
    reason::NTuple{EXCEPTION_REASON_LEN, UInt8}
    # the device-side stack trace, captured at debug level >= 2
    num_frames::Int32
    frames::NTuple{EXCEPTION_MAX_FRAMES, ExceptionFrame_st}

    ExceptionInfo_st() = new(
        0, 0, 0, (0, 0, 0, 0), (0, 0, 0, 0),
        ntuple(_ -> 0x00, EXCEPTION_NAME_LEN),
        ntuple(_ -> 0x00, EXCEPTION_REASON_LEN),
        0, ntuple(_ -> ExceptionFrame_st(), EXCEPTION_MAX_FRAMES)
    )
end

# Access fields through typed device pointers derived from the host struct layout.
const ExceptionInfo = LLVMPtr{UInt8, AS.CrossWorkgroup}

const EXCEPTION_INFO_OFFSETS = NamedTuple{fieldnames(ExceptionInfo_st)}(
    Tuple(Int(fieldoffset(ExceptionInfo_st, i)) for i in 1:fieldcount(ExceptionInfo_st))
)
const EXCEPTION_FRAME_OFFSETS = NamedTuple{fieldnames(ExceptionFrame_st)}(
    Tuple(Int(fieldoffset(ExceptionFrame_st, i)) for i in 1:fieldcount(ExceptionFrame_st))
)
const EXCEPTION_FRAME_SIZE = sizeof(ExceptionFrame_st)

@inline info_field(info::ExceptionInfo, ::Val{field}) where {field} =
    reinterpret(
    LLVMPtr{fieldtype(ExceptionInfo_st, field), AS.CrossWorkgroup},
    info + getfield(EXCEPTION_INFO_OFFSETS, field)
)
@inline frame_field(frame::ExceptionInfo, ::Val{field}) where {field} =
    reinterpret(
    LLVMPtr{fieldtype(ExceptionFrame_st, field), AS.CrossWorkgroup},
    frame + getfield(EXCEPTION_FRAME_OFFSETS, field)
)

# a byte pointer to the `idx`-th frame (1-based) in the mailbox's frame array
@inline frame_pointer(info::ExceptionInfo, idx) =
    info + EXCEPTION_INFO_OFFSETS.frames + (idx - 1) * EXCEPTION_FRAME_SIZE

# the mailbox address as carried by the kernel state; null when no mailbox was allocated
@inline exception_info() = reinterpret(ExceptionInfo, kernel_state().exception_info)
@inline has_exception_info(info::ExceptionInfo) = info != reinterpret(ExceptionInfo, 0)

# Claims must cover all work-groups on the device. SPIRVIntrinsics' public atomic helpers
# use work-group scope, which is not enough here.
@inline function atomic_claim_device!(p::LLVMPtr{Int32, AS.CrossWorkgroup})
    return @builtin_ccall(
        "__spirv_AtomicCompareExchange", Int32,
        (LLVMPtr{Int32, AS.CrossWorkgroup}, UInt32, UInt32, UInt32, Int32, Int32),
        p, UInt32(Scope.Device),
        UInt32(MemorySemantics.CrossWorkgroupMemory | MemorySemantics.AcquireRelease),
        UInt32(MemorySemantics.CrossWorkgroupMemory | MemorySemantics.Acquire),
        Int32(1), Int32(0)
    )
end

@inline function atomic_store_device!(p::LLVMPtr{Int32, AS.CrossWorkgroup}, val::Int32)
    @builtin_ccall(
        "__spirv_AtomicStore", Cvoid,
        (LLVMPtr{Int32, AS.CrossWorkgroup}, UInt32, UInt32, Int32),
        p, UInt32(Scope.Device),
        UInt32(MemorySemantics.CrossWorkgroupMemory | MemorySemantics.Release), val
    )
    return
end

# Store literal text as words to avoid a device-side copy loop. Name and reason buffers
# are 8-byte aligned and sized; zero-padding supplies the terminator. Keep this out of line
# because GPUCompiler inlines throwing functions into every call site.
@generated function store_string!(
        dest::LLVMPtr{NTuple{N, UInt8}, AS.CrossWorkgroup},
        ::Val{str}
    ) where {N, str}
    bytes = collect(codeunits(String(str)))
    n = min(length(bytes), N - 1)
    padded = bytes[1:n]
    push!(padded, 0x00)                              # null terminator
    while length(padded) % sizeof(UInt64) != 0       # zero-pad to a word boundary
        push!(padded, 0x00)
    end
    exprs = Expr[
        Expr(:meta, :noinline),
        :(base = reinterpret(LLVMPtr{UInt64, AS.CrossWorkgroup}, dest)),
    ]
    for w in 0:(length(padded) ÷ sizeof(UInt64) - 1)
        word = UInt64(0)
        for b in 0:(sizeof(UInt64) - 1)              # little-endian pack
            word |= UInt64(padded[w * sizeof(UInt64) + b + 1]) << (8 * b)
        end
        push!(exprs, :(unsafe_store!(base + $(w * sizeof(UInt64)), $word)))
    end
    push!(exprs, :(return nothing))
    return Expr(:block, exprs...)
end

# Runtime strings (GPUCompiler's names and frames) need a bounded byte copy.
@noinline function store_cstring!(
        dest::LLVMPtr{NTuple{N, UInt8}, AS.CrossWorkgroup},
        src::LLVMPtr{Cchar, AS.CrossWorkgroup}
    ) where {N}
    base = reinterpret(LLVMPtr{UInt8, AS.CrossWorkgroup}, dest)
    i = 0
    while i < N - 1
        c = unsafe_load(src + i) % UInt8
        c == 0x00 && break
        unsafe_store!(base + i, c)
        i += 1
    end
    unsafe_store!(base + i, 0x00)
    return
end

# Fold away all reporting at level 0. Only the owner may write details; level 2 allows
# that work-item to re-enter for the separate name and backtrace callbacks.
@inline lock_output!(info) = kernel_debug_level() < 1 ? false : claim_output!(info)

# Keep the claim out of line to limit code growth in kernels with many checked accesses.
@noinline function claim_output!(info)
    has_exception_info(info) || return false
    lock_ptr = info_field(info, Val(:output_lock))
    claimed = atomic_claim_device!(lock_ptr)
    # At level 1 all details are recorded in one call; only level 2 needs re-entry.
    kernel_debug_level() < 2 && return claimed == 0

    item = (
        get_local_id(1) % UInt64, get_local_id(2) % UInt64, get_local_id(3) % UInt64,
        UInt64(0),
    )
    group = (
        get_group_id(1) % UInt64, get_group_id(2) % UInt64, get_group_id(3) % UInt64,
        UInt64(0),
    )
    launch_id = kernel_state().launch_id
    if claimed == 0
        unsafe_store!(info_field(info, Val(:launch_id)), launch_id)
        unsafe_store!(info_field(info, Val(:work_item)), item)
        unsafe_store!(info_field(info, Val(:work_group)), group)
        # Publish the owner before another work-item reads it. A losing claimant must
        # never spin: the owner might be an inactive lane of the same subgroup.
        atomic_store_device!(lock_ptr, Int32(2))
        return true
    end
    claimed == 2 || return false
    # The acquire above pairs with publication. Coordinates alone are insufficient:
    # another kernel can use the same work-item and work-group ids.
    return unsafe_load(info_field(info, Val(:launch_id))) == launch_id &&
        unsafe_load(info_field(info, Val(:work_item))) == item &&
        unsafe_load(info_field(info, Val(:work_group))) == group
end

# One helper per name/reason pair keeps each throw site to a single cold call.
@noinline function record_exception!(info, ::Val{name}, ::Val{reason}) where {name, reason}
    if lock_output!(info)
        store_string!(info_field(info, Val(:name)), Val(name))
        store_string!(info_field(info, Val(:reason)), Val(reason))
    end
    return
end

"""
    @gputhrow name reason

Record the literal exception `name` and `reason` in the exception mailbox and throw, so
that the host can report the failure when it next synchronizes. Both strings must be
literals; use it in place of `throw` on device code paths.
"""
macro gputhrow(name::String, reason::String)
    name_q = QuoteNode(Symbol(name))
    reason_q = QuoteNode(Symbol(reason))
    return quote
        # the gate folds to a constant, so `-g0` kernels don't even carry the call
        if kernel_debug_level() >= 1
            record_exception!(exception_info(), Val($name_q), Val($reason_q))
        end
        throw(nothing)
    end
end

function signal_exception()
    info = exception_info()
    # All faulting work-items signal, including level 0 kernels that do not claim a record.
    # Kernel completion makes both the status and any recorded details visible to the host.
    if has_exception_info(info)
        atomic_store_device!(info_field(info, Val(:status)), Int32(1))
    end
    return
end

# Preserve a quirk's more precise literal name. Copy GPUCompiler's runtime strings only
# at level 2, keeping data-dependent copy loops out of ordinary level 1 throw paths.
function report_exception(ex)
    if kernel_debug_level() >= 2
        info = exception_info()
        name = info_field(info, Val(:name))
        if lock_output!(info) &&
                unsafe_load(reinterpret(LLVMPtr{UInt8, AS.CrossWorkgroup}, name)) == 0x00
            store_cstring!(name, ex)
        end
    end
    return
end
report_exception_name(ex) = report_exception(ex)

# GPUCompiler reports each stack frame (recovered from debug info) at debug level >= 2.
function report_exception_frame(idx, func, file, line)
    info = exception_info()
    if lock_output!(info) && 1 <= idx <= EXCEPTION_MAX_FRAMES
        frame = frame_pointer(info, idx)
        store_cstring!(frame_field(frame, Val(:func)), func)
        store_cstring!(frame_field(frame, Val(:file)), file)
        unsafe_store!(frame_field(frame, Val(:line)), Int32(line))
        unsafe_store!(info_field(info, Val(:num_frames)), Int32(idx))
    end
    return
end

report_oom(sz) = return

malloc(sz) = C_NULL

## kernel state

struct KernelState
    random_seed::UInt32

    # address of the device-side exception mailbox (`ExceptionInfo_st`); zero when no
    # mailbox is available. carried as an integer rather than a pointer: SPIR-V validation
    # rejects pointer-typed fields in by-value kernel arguments.
    exception_info::UInt64
    launch_id::UInt64
end

@inline @generated kernel_state() = GPUCompiler.kernel_state_value(KernelState)

## intrinsics for adding and accessing additional kernel arguments

# The amount of local shared memory we need for storing RNG state is determined
# dynamically at kernel launch time, so needs to be passed as additional arguments
# to the kernel.
# We define intrinsics that get transformed into additional kernel arguments which
# then get propagated across function calls to the caller.

function additional_arg_intr(mod::LLVM.Module, T_state, name)
    state_intr = if haskey(functions(mod), "julia.opencl.$name")
        functions(mod)["julia.opencl.$name"]
    else
        LLVM.Function(mod, "julia.opencl.$name", LLVM.FunctionType(T_state))
    end
    push!(function_attributes(state_intr), EnumAttribute("readnone", 0))

    return state_intr
end

# run-time equivalent
function additional_arg_value(state, name)
    return @dispose ctx = Context() begin
        T_state = convert(LLVMType, state)

        # create function
        llvm_f, _ = create_function(T_state)
        mod = LLVM.parent(llvm_f)

        # get intrinsic
        state_intr = additional_arg_intr(mod, T_state, name)
        state_intr_ft = function_type(state_intr)

        # generate IR
        @dispose builder = IRBuilder() begin
            entry = BasicBlock(llvm_f, "entry")
            position!(builder, entry)

            val = call!(builder, state_intr_ft, state_intr, Value[], name)

            ret!(builder, val)
        end

        call_function(llvm_f, state)
    end
end

for name in [:random_keys, :random_counters]
    @eval @inline @generated $name() =
        additional_arg_value(LLVMPtr{UInt32, AS.Workgroup}, $(String(name)))
end
