import KernelInterface as KI
using Random

# Counts every work-item at the element it identifies, computed from the group and local
# ids, so that a mix-up between group counts and group sizes shows.
function launch_kernel(arr)
    l = KI.get_local_id()
    g = KI.get_group_id()
    s = KI.get_local_size()
    i = (g.x - 1) * s.x + l.x
    j = (g.y - 1) * s.y + l.y
    k = (g.z - 1) * s.z + l.z
    if i <= size(arr, 1) && j <= size(arr, 2) && k <= size(arr, 3)
        @inbounds arr[i, j, k] += 1
    end
    return
end

struct KernelData
    global_size::Int
    global_id::Int
    local_size::Int
    local_id::Int
    num_groups::Int
    group_id::Int
end
function test_interface_kernel(results)
    i = KI.get_global_id().x

    if i <= length(results)
        @inbounds results[i] = KernelData(
            KI.get_global_size().x,
            KI.get_global_id().x,
            KI.get_local_size().x,
            KI.get_local_id().x,
            KI.get_num_groups().x,
            KI.get_group_id().x
        )
    end
    return
end
# Local arrays, written by one work-item and read back by another one after a barrier.
# `c` has the same type and size as `a`, but is a different call site, so different memory.
function localmem_kernel(out, ::Val{N}) where {N}
    i = KI.get_local_id().x
    a = KI.localmemory(Int32, N)
    b = KI.localmemory(Int32, (2, N))
    c = KI.localmemory(Int32, N)
    @inbounds begin
        a[i] = i
        b[1, i] = -i
        b[2, i] = 2i
        c[i] = 3i
    end
    KI.barrier()
    j = N - i + 1
    gid = KI.get_global_id().x
    @inbounds begin
        out[gid, 1] = a[j]
        out[gid, 2] = b[1, j]
        out[gid, 3] = b[2, j]
        out[gid, 4] = c[j]
    end
    return
end

# Every work-item writes global memory, and reads another work-item's write after a barrier.
function global_barrier_kernel(scratch, out)
    n = KI.get_local_size().x
    base = (KI.get_group_id().x - 1) * n
    i = KI.get_local_id().x
    @inbounds scratch[base + i] = base + i
    KI.barrier()
    @inbounds out[base + i] = scratch[base + n - i + 1]
    return
end

# The interface documents a concrete return type for each device-side function;
# these kernels record whether the backend honors them.
const WorkItemNT{T} = @NamedTuple{x::T, y::T, z::T}

function typecheck_kernel(results)
    @inbounds begin
        results[1] = KI.get_global_size() isa WorkItemNT{Int}
        results[2] = KI.get_global_id() isa WorkItemNT{Int}
        results[3] = KI.get_local_size() isa WorkItemNT{Int}
        results[4] = KI.get_local_id() isa WorkItemNT{Int}
        results[5] = KI.get_num_groups() isa WorkItemNT{Int}
        results[6] = KI.get_group_id() isa WorkItemNT{Int}
    end
    return
end

# The indexing queries take an element type; the result must use it.
function typed_typecheck_kernel(results, ::Type{T}) where {T}
    @inbounds begin
        results[1] = KI.get_global_size(T) isa WorkItemNT{T}
        results[2] = KI.get_global_id(T) isa WorkItemNT{T}
        results[3] = KI.get_local_size(T) isa WorkItemNT{T}
        results[4] = KI.get_local_id(T) isa WorkItemNT{T}
        results[5] = KI.get_num_groups(T) isa WorkItemNT{T}
        results[6] = KI.get_group_id(T) isa WorkItemNT{T}
    end
    return
end

# Records the typed indexing queries for every work-item, so the host can check
# that they agree with the default `Int` form across all three dimensions.
# `results` is `(work-items, 18)`: one row per work-item, holding the `x`, `y`
# and `z` components of each of the six queries in turn.
function typed_index_kernel(results, ::Type{T}) where {T}
    i, j, k = KI.get_global_id(T)
    ni, nj, _ = KI.get_global_size(T)
    lin = (k - one(T)) * ni * nj + (j - one(T)) * ni + i

    if lin <= size(results, 1)
        vals = (
            KI.get_global_size(T)..., KI.get_global_id(T)..., KI.get_local_size(T)...,
            KI.get_local_id(T)..., KI.get_num_groups(T)..., KI.get_group_id(T)...,
        )
        for q in 1:18
            @inbounds results[lin, q] = vals[q]
        end
    end
    return
end

# Records the `x` components of the typed queries in a type too small for the launch,
# indexed by the (untyped) global id.
function wrapping_index_kernel(results, ::Type{T}) where {T}
    i = KI.get_global_id().x
    if i <= size(results, 1)
        @inbounds begin
            results[i, 1] = KI.get_global_id(T).x
            results[i, 2] = KI.get_global_size(T).x
            results[i, 3] = KI.get_local_id(T).x
            results[i, 4] = KI.get_local_size(T).x
            results[i, 5] = KI.get_group_id(T).x
            results[i, 6] = KI.get_num_groups(T).x
        end
    end
    return
end

struct SubgroupData
    sub_group_size::Int
    max_sub_group_size::Int
    num_sub_groups::Int
    sub_group_id::Int
    sub_group_local_id::Int
end
function test_subgroup_kernel(results)
    l = KI.get_local_id()
    s = KI.get_local_size()
    i = ((l.z - 1) * s.y + (l.y - 1)) * s.x + l.x + (KI.get_group_id().x - 1) * s.x * s.y * s.z

    if i <= length(results)
        @inbounds results[i] = SubgroupData(
            KI.get_sub_group_size(),
            KI.get_max_sub_group_size(),
            KI.get_num_sub_groups(),
            KI.get_sub_group_id(),
            KI.get_sub_group_local_id()
        )
    end
    return
end

# Combine a value per sub-group through local memory, as reductions do: the first lane of
# every sub-group stores its size, and the first work-item adds up `get_num_sub_groups()`
# of them. `N` is the work-group size, which bounds the number of sub-groups.
function subgroup_combine_kernel(out, ::Val{N}) where {N}
    partial = KI.localmemory(Int32, N)
    if KI.get_sub_group_local_id() == 1
        @inbounds partial[KI.get_sub_group_id()] = KI.get_sub_group_size()
    end
    KI.barrier()
    l = KI.get_local_id()
    if l.x == 1 && l.y == 1 && l.z == 1
        total = Int32(0)
        for i in 1:KI.get_num_sub_groups()
            @inbounds total += partial[i]
        end
        @inbounds out[KI.get_group_id().x] = total
    end
    return
end

function subgroup_typecheck_kernel(results, val::T) where {T}
    # uniformly executed by the whole sub-group, as `shfl_down` requires
    shuffled = KI.shfl_down(val, 1)
    if KI.get_sub_group_local_id() == 1
        @inbounds begin
            results[1] = KI.get_sub_group_size() isa Int
            results[2] = KI.get_max_sub_group_size() isa Int
            results[3] = KI.get_num_sub_groups() isa Int
            results[4] = KI.get_sub_group_id() isa Int
            results[5] = KI.get_sub_group_local_id() isa Int
            results[6] = shuffled isa T
            results[7] = KI.get_sub_group_size(UInt32) isa UInt32
            results[8] = KI.get_max_sub_group_size(Int32) isa Int32
            results[9] = KI.get_num_sub_groups(UInt32) isa UInt32
            results[10] = KI.get_sub_group_id(Int32) isa Int32
            results[11] = KI.get_sub_group_local_id(UInt32) isa UInt32
        end
    end
    return
end

function shfl_down_test_kernel(a, b, ::Val{N}) where {N}
    idx = KI.get_sub_group_local_id()

    val = a[idx]

    # the result of shuffling from a lane past the end is unspecified, so don't add it
    offset = 1
    while offset < N
        shuffled = KI.shfl_down(val, offset)
        if idx + offset <= N
            val += shuffled
        end
        offset <<= 1
    end

    KI.sub_group_barrier()

    if idx == 1
        b[idx] = val
    end
    return
end

# Every lane shuffles its lane id down by `offset`; `out` gets the lane id, the sub-group
# size and the result.
function shfl_down_lanes_kernel(out, ::Type{T}, offset) where {T}
    lane = KI.get_sub_group_local_id()
    shuffled = KI.shfl_down(T(lane), offset)
    i = KI.get_global_id().x
    @inbounds begin
        out[i, 1] = lane
        out[i, 2] = KI.get_sub_group_size()
        out[i, 3] = shuffled
    end
    return
end

# Every lane reads the lane `shift` further, wrapping around: the rotation of a tile.
function shfl_rotate_kernel(out, a::AbstractArray{T}, shift) where {T}
    lane = KI.get_sub_group_local_id()
    width = KI.get_sub_group_size()
    val = @inbounds a[lane]
    @inbounds out[lane] = KI.shfl(val, mod1(lane + shift, width))
    return
end

function shfl_up_lanes_kernel(out, ::Type{T}, offset) where {T}
    lane = KI.get_sub_group_local_id()
    shuffled = KI.shfl_up(T(lane), offset)
    @inbounds out[lane] = shuffled
    return
end

# An all-reduce with a butterfly: every lane gets the sum of the sub-group. `N` is the
# width, and the work-group is one sub-group.
function shfl_xor_sum_kernel(out, a, ::Val{N}) where {N}
    lane = KI.get_sub_group_local_id()
    val = @inbounds a[lane]
    mask = N >> 1
    while mask > 0
        val += KI.shfl_xor(val, mask)
        mask >>= 1
    end
    @inbounds out[lane] = val
    return
end

struct ShuffleStruct
    a::Float32
    b::Int64
    c::NTuple{3, Int32}
end

function shfl_struct_kernel(out, a)
    lane = KI.get_sub_group_local_id()
    width = KI.get_sub_group_size()
    val = @inbounds a[lane]
    @inbounds out[lane] = KI.shfl(val, mod1(lane + 1, width))
    return
end

function vote_kernel(out, pred)
    lane = KI.get_sub_group_local_id()
    p = @inbounds pred[lane]
    any = KI.sub_group_any(p)
    all = KI.sub_group_all(p)
    ballot = KI.sub_group_ballot(p)
    @inbounds begin
        out[lane, 1] = any
        out[lane, 2] = all
        out[lane, 3] = ballot
        out[lane, 4] = KI.get_max_sub_group_size()
    end
    return
end

# Each work-item shuffles its value with `op` (`shfl`, `shfl_down`, `shfl_up`, `shfl_xor`) and
# a `width`, recording its lane.
function shfl_width_kernel(out, lanes, a, op, arg, width)
    lane = KI.get_sub_group_local_id()
    val = @inbounds a[lane]
    @inbounds out[lane] = op(val, arg, width)
    @inbounds lanes[lane] = lane
    return
end

# `pred` determines which values are equal, `vals` gives the values
function match_any_kernel(out, vals)
    lane = KI.get_sub_group_local_id()
    @inbounds out[lane] = KI.sub_group_match_any(vals[lane])
    return
end

# a work-group of `length(a)` work-items, which is a single (possibly partial) sub-group
function reduce_scan_kernel(red, scan, lanes, a, op)
    i = KI.get_local_id().x
    val = @inbounds a[i]
    r = KI.sub_group_reduce(op, val)
    s = KI.sub_group_scan(op, val)
    @inbounds begin
        red[i] = r
        scan[i] = s
        lanes[i] = KI.get_sub_group_local_id()
    end
    return
end

# primitive types that no backend supports natively, shuffled as `UInt32` words
primitive type Bits64 64 end
primitive type Bits16 16 end
Bits64(x::Integer) = reinterpret(Bits64, x % UInt64)
Bits16(x::Integer) = reinterpret(Bits16, x % UInt16)

# the values come from a divergent branch, like for the padding work-items of a `@kernel`
function reduce_divergent_kernel(red, scan, a, m)
    i = KI.get_local_id().x
    val = i <= m ? (@inbounds a[i]) : zero(eltype(a))
    r = KI.sub_group_reduce(+, val)
    s = KI.sub_group_scan(+, val)
    @inbounds red[i] = r
    @inbounds scan[i] = s
    return
end

# the votes within segments of `width` lanes; a work-group of `length(pred)` work-items, i.e. a
# single (possibly partial) sub-group
function segmented_vote_kernel(out, lanes, pred, vals, width)
    i = KI.get_local_id().x
    p = @inbounds pred[i]
    any = KI.sub_group_any(p, width)
    all = KI.sub_group_all(p, width)
    ballot = KI.sub_group_ballot(p, width)
    match = KI.sub_group_match_any(@inbounds(vals[i]), width)
    @inbounds begin
        out[i, 1] = any
        out[i, 2] = all
        out[i, 3] = ballot
        out[i, 4] = match
        lanes[i] = KI.get_sub_group_local_id()
    end
    return
end

# the sub-group and lane of every work-item, by its linear index (x fastest)
function sub_group_layout_kernel(out)
    l = KI.get_local_id()
    sz = KI.get_local_size()
    lin = l.x + (l.y - 1) * sz.x + (l.z - 1) * sz.x * sz.y
    @inbounds begin
        out[1, lin] = KI.get_sub_group_id()
        out[2, lin] = KI.get_sub_group_local_id()
        out[3, lin] = KI.get_sub_group_size()
    end
    return
end

# reductions and scans in a 1-D work-group of several sub-groups, the last one partial
function reduce_scan_multi_kernel(red, scan, sgs, a)
    i = KI.get_local_id().x
    val = @inbounds a[i]
    r = KI.sub_group_reduce(+, val)
    s = KI.sub_group_scan(+, val)
    @inbounds begin
        red[i] = r
        scan[i] = s
        sgs[i] = KI.get_sub_group_id()
    end
    return
end

struct FallbackStruct
    flag::Bool
    c::Char
    x::Bits64
    limbs::NTuple{4, Float64}
end

# Every lane writes local and global memory, and reads another lane's write after a
# sub-group barrier. `N` is the sub-group width; the work-group is one sub-group.
function sub_group_barrier_kernel(scratch, out, ::Val{N}) where {N}
    lane = KI.get_sub_group_local_id()
    other = mod1(lane + 1, KI.get_sub_group_size())
    lm = KI.localmemory(Int32, N)
    @inbounds lm[lane] = lane
    @inbounds scratch[lane] = -lane
    KI.sub_group_barrier()
    @inbounds out[lane, 1] = lm[other]
    @inbounds out[lane, 2] = scratch[other]
    return
end

# a kernel whose callable captures an array, compiled but not launched yet
function captured_array_kernel(backend, AT, out)
    a = AT(Int32[42])
    kernel = KI.@launch backend launch = false (() -> (@inbounds out[1] = a[1]; nothing))()
    return kernel, WeakRef(a)
end

# an associative but not commutative operator: the composition of affine maps
compose_affine(f, g) = (g[1] * f[1], g[1] * f[2] + g[2])
# the (value, index) of the smallest value, the first one of equal values
argmin_op(x, y) = ifelse(y[1] < x[1], y, x)

# The sub-group operations that are built on the shuffles and votes. Separate functions, so
# that `interface_testsuite` doesn't get too large to compile.

# rotating values of type `T` by one lane, `f(i)` giving the value of lane `i`
function shfl_type_testsuite(backend, AT, sg_size, ::Type{T}, f) where {T}
    a = T[f(i) for i in 1:sg_size]
    out = AT(fill(f(0), sg_size))
    KI.@launch backend workgroupsize = sg_size shfl_rotate_kernel(out, AT(a), 1)
    KI.synchronize(backend)
    @test Array(out) == circshift(a, -1)
    return
end

# `ref(lane, arg, width)` is the lane that `op(val, arg, width)` reads from
function shfl_width_testsuite(backend, AT, sg_size, op, ref, width, arg)
    a = Int32.(1:sg_size) .* Int32(10)
    out = AT(zeros(Int32, sg_size))
    lanes = AT(zeros(Int, sg_size))
    KI.@launch backend workgroupsize = sg_size shfl_width_kernel(out, lanes, AT(a), op, arg, width)
    KI.synchronize(backend)
    out, lanes = Array(out), Array(lanes)
    @test all(i -> out[i] == a[ref(lanes[i], arg, width)], 1:sg_size)
    return
end

shfl_width_ref(l, arg, w) = (l - 1) ÷ w * w + mod1(arg, w)
shfl_down_width_ref(l, arg, w) = (l - 1) % w + arg < w ? l + arg : l
shfl_up_width_ref(l, arg, w) = (l - 1) % w >= arg ? l - arg : l
shfl_xor_width_ref(l, arg, w) = (((l - 1) ⊻ arg) ÷ w == (l - 1) ÷ w) ? ((l - 1) ⊻ arg) + 1 : l

function match_any_testsuite(backend, AT, sg_size, vals)
    out = AT(zeros(UInt64, sg_size))
    KI.@launch backend workgroupsize = sg_size match_any_kernel(out, AT(vals))
    KI.synchronize(backend)
    out = Array(out)
    for i in 1:sg_size
        expected = UInt64(0)
        for j in 1:sg_size
            vals[j] === vals[i] && (expected |= UInt64(1) << (j - 1))
        end
        @test out[i] == expected
    end
    return
end

# a work-group of `length(a)` work-items, i.e. a single sub-group, possibly partial
function reduce_scan_testsuite(backend, AT, op, a)
    n = length(a)
    red = AT(similar(a))
    scan = AT(similar(a))
    lanes = AT(zeros(Int, n))
    KI.@launch backend workgroupsize = n reduce_scan_kernel(red, scan, lanes, AT(a), op)
    KI.synchronize(backend)
    red, scan, lanes = Array(red), Array(scan), Array(lanes)
    # in the order of the lanes
    order = sortperm(lanes)
    # `isequal`, as the results may be NaN
    @test all(isequal(foldl(op, a[order])), red)
    @test all(i -> isequal(scan[order[i]], foldl(op, a[order[1:i]])), 1:n)
    return
end

function reduce_divergent_testsuite(backend, AT, sg_size, ::Type{T}) where {T}
    m = max(sg_size - 5, 1)
    a = T.(rand(1:20, sg_size))
    red = AT(zeros(T, sg_size))
    scan = AT(zeros(T, sg_size))
    KI.@launch backend workgroupsize = sg_size reduce_divergent_kernel(red, scan, AT(a), m)
    KI.synchronize(backend)
    # the sum doesn't depend on the order of the lanes
    @test all(==(sum(a[1:m])), Array(red))
    @test maximum(Array(scan)) == sum(a[1:m])
    return
end

function segmented_vote_testsuite(backend, AT, n, width)
    pred = [rand(Bool) for _ in 1:n]
    # make some segments all true and some all false
    for i in 1:n
        seg = (i - 1) ÷ width
        seg % 3 == 1 && (pred[i] = true)
        seg % 3 == 2 && (pred[i] = false)
    end
    vals = Int32.(rand(1:3, n))
    out = AT(zeros(UInt64, n, 4))
    lanes = AT(zeros(Int, n))
    KI.@launch backend workgroupsize = n segmented_vote_kernel(out, lanes, AT(pred), AT(vals), width)
    KI.synchronize(backend)
    out, lanes = Array(out), Array(lanes)
    by_lane = zeros(Int, maximum(lanes))
    for i in 1:n
        by_lane[lanes[i]] = i
    end
    for i in 1:n
        base = (lanes[i] - 1) ÷ width * width
        # the work-items of the segment, by their position in the segment
        seg = [(l - base, by_lane[l]) for l in (base + 1):min(base + width, length(by_lane)) if by_lane[l] != 0]
        ballot = UInt64(0)
        match = UInt64(0)
        for (k, j) in seg
            pred[j] && (ballot |= UInt64(1) << (k - 1))
            vals[j] == vals[i] && (match |= UInt64(1) << (k - 1))
        end
        @test out[i, 1] == any(pred[j] for (_, j) in seg)
        @test out[i, 2] == all(pred[j] for (_, j) in seg)
        @test out[i, 3] == ballot
        @test out[i, 4] == match
    end
    return
end

# 1-D work-groups and ones whose x extent is a multiple of the sub-group width form
# sub-groups from consecutive work-items, x fastest
function sub_group_layout_testsuite(backend, AT, sg_size, fits)
    shapes = ((sg_size + 5,), (3 * sg_size,), (sg_size, 4), (2 * sg_size, 2), (sg_size, 2, 2))
    for dims in shapes
        n = prod(dims)
        out = AT(zeros(Int, 3, n))
        kernel = KI.@launch backend launch = false sub_group_layout_kernel(out)
        if !fits(kernel, dims)
            @test_skip "work-groups of $dims work-items"
            continue
        end
        kernel(out; workgroupsize = dims)
        KI.synchronize(backend)
        out = Array(out)
        @testset "$dims" begin
            @test out[1, :] == [(lin - 1) ÷ sg_size + 1 for lin in 1:n]
            @test out[2, :] == [(lin - 1) % sg_size + 1 for lin in 1:n]
            @test out[3, :] == [min(sg_size, n - (lin - 1) ÷ sg_size * sg_size) for lin in 1:n]
        end
    end
    return
end

function reduce_scan_multi_testsuite(backend, AT, sg_size, ::Type{T}) where {T}
    n = 2 * sg_size + 5
    a = T.(rand(1:20, n))
    red, scan, sgs = AT(zeros(T, n)), AT(zeros(T, n)), AT(zeros(Int, n))
    kernel = KI.@launch backend launch = false reduce_scan_multi_kernel(red, scan, sgs, AT(a))
    n <= KI.max_work_group_size(kernel) || return
    kernel(red, scan, sgs, AT(a); workgroupsize = n)
    KI.synchronize(backend)
    red, scan, sgs = Array(red), Array(scan), Array(sgs)
    for i in 1:n
        group = findall(==(sgs[i]), sgs)
        @test red[i] == sum(a[group])
    end
    # 1-D work-groups form sub-groups from consecutive work-items
    @test scan == [sum(a[((i - 1) ÷ sg_size * sg_size + 1):i]) for i in 1:n]
    return
end

function subgroup_communication_testsuite(backend::KI.Backend, AT, sg_size)
    @testset "shuffles of other types" begin
        # primitive types that backends need not support natively are shuffled as words
        if KI.supports_shuffle(backend, UInt32)
            @test KI.supports_shuffle(backend, Bool)
            @test KI.supports_shuffle(backend, Char)
            @test KI.supports_shuffle(backend, Bits64)
            @test KI.supports_shuffle(backend, Bits16)
            @test KI.supports_shuffle(backend, FallbackStruct)
        end
        @test !KI.supports_shuffle(backend, Ref{Int})
        @testset "Bool" begin
            KI.supports_shuffle(backend, Bool) && shfl_type_testsuite(backend, AT, sg_size, Bool, isodd)
        end
        @testset "Char" begin
            KI.supports_shuffle(backend, Char) &&
                shfl_type_testsuite(backend, AT, sg_size, Char, i -> Char(0x0001F600 + i))
        end
        @testset "Bits64" begin
            KI.supports_shuffle(backend, Bits64) &&
                shfl_type_testsuite(backend, AT, sg_size, Bits64, i -> Bits64((UInt64(i) << 40) - i))
        end
        @testset "Bits16" begin
            KI.supports_shuffle(backend, Bits16) &&
                shfl_type_testsuite(backend, AT, sg_size, Bits16, i -> Bits16(0xa000 + i))
        end
        @testset "struct" begin
            KI.supports_shuffle(backend, FallbackStruct) && shfl_type_testsuite(
                backend, AT, sg_size, FallbackStruct,
                i -> FallbackStruct(isodd(i), Char(64 + i), Bits64(-i), (i, -i, 1 / i, 2.0^i))
            )
        end
    end

    KI.supports_shuffle(backend, Int32) || return

    @testset "shuffles with a width" begin
        for w in (1, 2, 4, 8, 16, 32, 64)
            (w <= sg_size && sg_size % w == 0) || continue
            for arg in unique((1, 3, w - 1, w + 2))
                @testset "width $w, $arg" begin
                    shfl_width_testsuite(backend, AT, sg_size, KI.shfl, shfl_width_ref, w, arg)
                    shfl_width_testsuite(backend, AT, sg_size, KI.shfl_down, shfl_down_width_ref, w, arg)
                    shfl_width_testsuite(backend, AT, sg_size, KI.shfl_up, shfl_up_width_ref, w, arg)
                    shfl_width_testsuite(backend, AT, sg_size, KI.shfl_xor, shfl_xor_width_ref, w, arg)
                end
            end
        end
    end

    if sg_size <= 64
        @testset "sub_group_match_any" begin
            match_any_testsuite(backend, AT, sg_size, Int32[i % 3 for i in 1:sg_size])
            # bitwise comparison, so NaN matches NaN, and -0.0 doesn't match 0.0
            match_any_testsuite(
                backend, AT, sg_size,
                Float32[isodd(i) ? NaN32 : (i % 4 == 0 ? -0.0f0 : 0.0f0) for i in 1:sg_size]
            )
        end
    end

    if sg_size <= 64
        @testset "votes with a width" begin
            for n in unique((sg_size, max(sg_size - 3, 1))), w in (1, 2, 4, 8, 16, 32, 64)
                (w <= sg_size && sg_size % w == 0) || continue
                @testset "$n work-items, width $w" begin
                    segmented_vote_testsuite(backend, AT, n, w)
                end
            end
        end
    end

    @testset "sub_group_reduce and sub_group_scan, several sub-groups" begin
        reduce_scan_multi_testsuite(backend, AT, sg_size, Int32)
        KI.supports_shuffle(backend, Float32) && reduce_scan_multi_testsuite(backend, AT, sg_size, Float32)
    end

    @testset "sub_group_reduce and sub_group_scan of divergent values" begin
        reduce_divergent_testsuite(backend, AT, sg_size, Int32)
        KI.supports_shuffle(backend, Float32) && reduce_divergent_testsuite(backend, AT, sg_size, Float32)
    end

    for n in unique((sg_size, max(sg_size - 3, 1)))
        @testset "sub_group_reduce and sub_group_scan, $n work-items" begin
            reduce_scan_testsuite(backend, AT, +, Int32.(rand(1:100, n)))
            # operators and types that backends may implement natively
            reduce_scan_testsuite(backend, AT, min, Int64.(rand(-100:100, n)))
            reduce_scan_testsuite(backend, AT, max, UInt32.(rand(1:100, n)))
            KI.supports_shuffle(backend, Float32) &&
                reduce_scan_testsuite(backend, AT, +, Float32.(rand(1:100, n)))
            KI.supports_shuffle(backend, Float32) &&
                reduce_scan_testsuite(backend, AT, max, Float32[i == 2 ? NaN32 : rand(1:100) for i in 1:n])
            reduce_scan_testsuite(
                backend, AT, compose_affine,
                [(Int32(rand((-1, 1, 2))), Int32(rand(-5:5))) for _ in 1:n]
            )
            reduce_scan_testsuite(backend, AT, argmin_op, [(Float32(rand(1:20)), Int32(i)) for i in 1:n])
        end
    end
    return
end

function interface_testsuite(backend::KI.Backend, AT)
    @testset "Launch parameters" begin
        # unequal group counts and sizes in every dimension, so that confusing them shows
        function run(dims; kwargs...)
            arr = KI.zeros(backend, Int32, dims)
            kernel = KI.@launch backend launch = false launch_kernel(arr)
            kernel(arr; kwargs...)
            KI.synchronize(backend)
            return Array(arr)
        end
        @test all(==(1), run((6, 1, 1); numgroups = 3, workgroupsize = 2))
        @test all(==(1), run((6, 1, 1); numgroups = (2,), workgroupsize = (3,)))
        @test all(==(1), run((6, 10, 1); numgroups = (2, 5), workgroupsize = (3, 2)))
        @test all(==(1), run((6, 10, 12); numgroups = (2, 5, 3), workgroupsize = (3, 2, 4)))

        # `ndrange` rounds up to whole groups, and doesn't mask the padding
        @test all(==(1), run((7, 5, 3); ndrange = (7, 5, 3), workgroupsize = (2, 3, 2)))
        @test all(==(1), run((7, 5, 3); ndrange = (7, 5, 3)))
        @test all(==(1), run((1000, 1, 1); ndrange = 1000))

        # defaults: one work-group of one work-item
        @test run((2, 2, 1)) == reshape(Int32[1, 0, 0, 0], 2, 2, 1)
        @test run((4, 1, 1); numgroups = 2) == reshape(Int32[1, 1, 0, 0], 4, 1, 1)
        @test run((4, 1, 1); workgroupsize = 2) == reshape(Int32[1, 1, 0, 0], 4, 1, 1)

        # nothing to launch
        @test all(==(0), run((2, 2, 2); ndrange = 0))
        @test all(==(0), run((2, 2, 2); ndrange = (2, 0)))
        @test all(==(0), run((2, 2, 2); numgroups = (2, 0, 2), workgroupsize = 2))

        # the global size is the padded ndrange
        results = AT(Vector{KernelData}(undef, 12))
        kernel = KI.@launch backend launch = false test_interface_kernel(results)
        kernel(results; ndrange = 10, workgroupsize = 4)
        KI.synchronize(backend)
        @test all(d -> d.global_size == 12 && d.num_groups == 3, Array(results))
    end

    @testset "Launch validation" begin
        arr = KI.zeros(backend, Int32, (1, 1, 1))
        kernel = KI.@launch backend launch = false launch_kernel(arr)
        max_items = KI.max_work_group_size(kernel)
        max_dims = KI.max_work_group_dims(backend)

        @test_throws ArgumentError kernel(arr; numgroups = (2, 2, 2, 2), workgroupsize = (2, 2, 2))
        @test_throws ArgumentError kernel(arr; numgroups = (2, 2, 2), workgroupsize = (2, 2, 2, 2))
        @test_throws ArgumentError kernel(arr; ndrange = (2, 2, 2, 2))
        @test_throws ArgumentError kernel(arr; ndrange = 4, numgroups = 2)
        @test_throws ArgumentError kernel(arr; workgroupsize = 0)
        @test_throws ArgumentError kernel(arr; workgroupsize = (2, 0))
        @test_throws ArgumentError kernel(arr; workgroupsize = -1)
        @test_throws ArgumentError kernel(arr; numgroups = -1)
        @test_throws ArgumentError kernel(arr; ndrange = (4, -1))
        @test_throws ArgumentError kernel(arr; ndrange = 4.0)
        @test_throws ArgumentError kernel(arr; ndrange = 4, max_work_group_size = 0)
        @test_throws ArgumentError kernel(arr; workgroupsize = max_items + 1)
        if max_dims[3] < max_items
            @test_throws ArgumentError kernel(arr; workgroupsize = (1, 1, max_dims[3] + 1))
        end
        KI.synchronize(backend)
        @test Array(arr) == zeros(Int32, 1, 1, 1)

        # other keywords are passed to the backend, which rejects the ones it doesn't know
        @test_throws Exception kernel(arr; this_is_not_a_launch_option = 1)
    end

    @testset "Launch limits" begin
        max_dims = KI.max_work_group_dims(backend)
        max_groups = KI.max_num_groups(backend)
        @test max_dims isa NTuple{3, Int} && all(>=(1), max_dims)
        @test max_groups isa NTuple{3, Int} && all(>=(1), max_groups)
        @test KI.max_work_group_size(backend) isa Int

        function fill_kernel(arr)
            i, j, k = KI.get_global_id()
            if i <= size(arr, 1) && j <= size(arr, 2) && k <= size(arr, 3)
                @inbounds arr[i, j, k] = 1.0f0
            end
            return
        end
        kernel = KI.@launch backend launch = false fill_kernel(AT(zeros(Float32, 1, 1, 1)))
        function fill_test(dims; kwargs...)
            arr = AT(zeros(Float32, dims))
            kernel(arr; kwargs...)
            KI.synchronize(backend)
            return all(Array(arr) .== 1)
        end

        max_items = KI.max_work_group_size(kernel)
        @test max_items isa Int && 1 <= max_items <= KI.max_work_group_size(backend)
        config = KI.launch_configuration(kernel)
        @test config isa @NamedTuple{workgroupsize::Int}
        @test 1 <= config.workgroupsize <= max_items
        @test KI.launch_configuration(kernel; max_work_group_size = 1).workgroupsize == 1
        # the recommendation is legal, whatever the size of the launch
        @testset "nitems = $nitems, max_work_group_size = $cap" for nitems in (nothing, 1, 1000, typemax(Int)),
                cap in (1, 7, typemax(Int))
            config = KI.launch_configuration(kernel; nitems, max_work_group_size = cap)
            @test 1 <= config.workgroupsize <= min(cap, max_items)
        end

        # automatically chosen workgroup sizes respect the per-dimension limits
        @testset "ndrange = $ndrange" for ndrange in ((1, 1, 5000), (1, 5000, 1), (1, 3, 2000))
            @test fill_test(ndrange; ndrange)
            @test fill_test(ndrange; ndrange, max_work_group_size = 7)
        end

        # the reported limits can be launched
        @testset "dimension $d" for d in 1:3
            items = min(max_dims[d], max_items)
            workgroupsize = ntuple(i -> i == d ? items : 1, 3)
            @test fill_test(workgroupsize; workgroupsize, numgroups = (1, 1, 1))

            # don't launch (practically) unlimited grids
            groups = min(max_groups[d], 2^16)
            numgroups = ntuple(i -> i == d ? groups : 1, 3)
            @test fill_test(numgroups; workgroupsize = (1, 1, 1), numgroups)
        end
    end

    @testset "Host return types" begin
        b = backend

        @test KI.supports_unified(b) isa Bool
        @test KI.supports_atomics(b) isa Bool
        @test KI.supports_float64(b) isa Bool
        @test KI.supports_subgroups(b) isa Bool
        @test KI.supports_shuffle(b, Float32) isa Bool
        @test KI.functional(b) isa Union{Missing, Bool}
        @test KI.multiprocessor_count(b) isa Int

        @test KI.device(b) isa Int
        @test KI.ndevices(b) isa Int

        arr = KI.allocate(b, Float32, 2)
        @test arr isa AT{Float32, 1}
        @test KI.zeros(b, Float32, 2) isa AT{Float32, 1}
        @test KI.ones(b, Float32, 2) isa AT{Float32, 1}
        @test KI.get_backend(arr) isa KI.Backend
        @test KI.device(b, arr) == KI.device(b)

        kernel = KI.@launch b launch = false test_interface_kernel(AT(Vector{KernelData}(undef, 1)))
        @test kernel isa KI.Kernel
        @test kernel.backend === b
    end

    @testset "Devices" begin
        b = backend
        dev = KI.device(b)
        @test 1 <= dev <= KI.ndevices(b)
        KI.device!(b, dev)
        @test KI.device(b) == dev
        @test_throws ArgumentError KI.device!(b, 0)
        @test_throws ArgumentError KI.device!(b, KI.ndevices(b) + 1)

        if KI.ndevices(b) > 1
            other = mod1(dev + 1, KI.ndevices(b))
            arr_dev = KI.ones(b, Float32, 4)
            try
                KI.device!(b, other)
                # the owner of an array doesn't change with the active device
                @test KI.device(b, arr_dev) == dev
                @test KI.device(b) == other
                arr = KI.ones(b, Float32, 4)
                @test KI.device(b, arr) == other
                @test Array(arr) == ones(Float32, 4)
            finally
                KI.device!(b, dev)
            end
            @test KI.device(b) == dev
        end
    end

    @testset "copyto!" begin
        b = backend
        host = rand(Float32, 16)
        dev = KI.allocate(b, Float32, 16)
        @test KI.copyto!(b, dev, host) === dev
        dev2 = KI.allocate(b, Float32, 16)
        @test KI.copyto!(b, dev2, dev) === dev2
        back = zeros(Float32, 16)
        @test KI.copyto!(b, back, dev2) === back
        KI.synchronize(b)
        @test back == host

        # arrays of different shapes copy by linear index
        dev3 = KI.allocate(b, Float32, (4, 4))
        KI.copyto!(b, dev3, host)
        KI.synchronize(b)
        @test Array(dev3) == reshape(host, 4, 4)

        # ordered with respect to kernels on the same queue
        arr = KI.zeros(b, Int32, (4, 1, 1))
        KI.@launch b numgroups = 1 workgroupsize = 4 launch_kernel(arr)
        result = zeros(Int32, 4)
        KI.copyto!(b, result, arr)
        KI.synchronize(b)
        @test result == ones(Int32, 4)

        @test_throws ArgumentError KI.copyto!(b, KI.allocate(b, Float32, 8), host)
        @test_throws ArgumentError KI.copyto!(b, zeros(Float32, 8), dev)
        @test_throws ArgumentError KI.copyto!(b, KI.allocate(b, Float32, 32), dev)
        KI.synchronize(b)
    end

    @testset "Device return types" begin
        results = KI.zeros(backend, Bool, 6)
        KI.@launch backend typecheck_kernel(results)
        KI.synchronize(backend)
        @test all(Array(results))

        @testset "$T" for T in (Int32, Int64, UInt32, UInt64)
            typed_results = KI.zeros(backend, Bool, 6)
            KI.@launch backend typed_typecheck_kernel(typed_results, T)
            KI.synchronize(backend)
            @test all(Array(typed_results))
        end
    end

    @testset "Typed indexing" begin
        workgroupsize = (2, 3, 2)
        numgroups = (3, 2, 4)
        N = prod(workgroupsize) * prod(numgroups)

        # `Int` is the reference: it is what the zero-argument form returns.
        function run_typed(::Type{T}) where {T}
            results = KI.zeros(backend, T, N, 18)
            KI.@launch backend workgroupsize = workgroupsize numgroups = numgroups typed_index_kernel(results, T)
            KI.synchronize(backend)
            return Array(results)
        end
        reference = run_typed(Int)

        global_size = workgroupsize .* numgroups
        @test all(eachrow(reference[:, 1:3]) .== Ref(collect(global_size)))
        @test all(eachrow(reference[:, 7:9]) .== Ref(collect(workgroupsize)))
        @test all(eachrow(reference[:, 13:15]) .== Ref(collect(numgroups)))
        # every global id is seen exactly once, and agrees with the group and local ids
        @test sort(Tuple.(eachrow(reference[:, 4:6]))) == sort(vec(Tuple.(CartesianIndices(global_size))))
        @test reference[:, 4:6] == (reference[:, 16:18] .- 1) .* reference[:, 7:9] .+ reference[:, 10:12]

        @testset "$T" for T in (Int32, UInt32, Int64, UInt64)
            typed = run_typed(T)
            @test typed isa AbstractMatrix{T}
            @test typed == reference
        end
    end

    @testset "Typed indexing wraps" begin
        # a type too small for the launch wraps around (like `x % T`) instead of throwing
        @testset "$T" for (T, workgroupsize, numgroups) in (
                (UInt8, 128, 3), (Int8, 128, 3), (Int16, 256, 160),
            )
            N = workgroupsize * numgroups
            results = KI.zeros(backend, T, N, 6)
            KI.@launch backend workgroupsize = workgroupsize numgroups = numgroups wrapping_index_kernel(results, T)
            KI.synchronize(backend)
            results = Array(results)
            ids = 1:N
            @test results[:, 1] == ids .% T
            @test all(==(N % T), results[:, 2])
            @test results[:, 3] == mod1.(ids, workgroupsize) .% T
            @test all(==(workgroupsize % T), results[:, 4])
            @test results[:, 5] == cld.(ids, workgroupsize) .% T
            @test all(==(numgroups % T), results[:, 6])
        end
    end

    @testset "Basic interface functionality" begin
        workgroupsize = 4
        numgroups = 3
        N = workgroupsize * numgroups
        results = AT(Vector{KernelData}(undef, N))
        kernel = KI.@launch backend launch = false test_interface_kernel(results)

        kernel(results; workgroupsize, numgroups)
        KI.synchronize(backend)

        host_results = Array(results)
        for (i, k_data) in enumerate(host_results)
            @test k_data.global_id == i
            @test k_data.global_size == N
            @test k_data.local_size == workgroupsize
            @test k_data.num_groups == numgroups
            @test k_data.group_id == div(i - 1, workgroupsize) + 1
            @test k_data.local_id == ((i - 1) % workgroupsize) + 1
        end
    end

    # The converted callable only holds pointers to the arrays it captures, so the kernel
    # has to keep the original alive (and a backend may need it at launch).
    @testset "Captured arrays" begin
        out = AT(Int32[0])
        kernel, captured = captured_array_kernel(backend, AT, out)
        GC.gc(true)
        @test captured.value !== nothing
        garbage = [AT(fill(Int32(7), 1)) for _ in 1:100]
        kernel()
        KI.synchronize(backend)
        @test Array(out) == Int32[42]
    end

    @testset "Local memory and barriers" begin
        N = 32
        groups = 3
        out = KI.zeros(backend, Int32, N * groups, 4)
        KI.@launch backend numgroups = groups workgroupsize = N localmem_kernel(out, Val(N))
        KI.synchronize(backend)
        out = Array(out)
        # each work-item sees what its mirror image wrote before the barrier, in both arrays
        expected = repeat(N:-1:1, groups)
        @test out[:, 1] == expected
        @test out[:, 2] == -expected
        @test out[:, 3] == 2 .* expected
        @test out[:, 4] == 3 .* expected

        scratch = KI.zeros(backend, Int, N * groups)
        out = KI.zeros(backend, Int, N * groups)
        KI.@launch backend numgroups = groups workgroupsize = N global_barrier_kernel(scratch, out)
        KI.synchronize(backend)
        @test Array(out) == vcat([(g * N) .+ (N:-1:1) for g in 0:(groups - 1)]...)
    end

    if KI.supports_subgroups(backend)
        sg_size = KI.sub_group_size(backend)
        max_dims = KI.max_work_group_dims(backend)
        # whether `kernel` can be launched with work-groups of size `dims`
        fits(kernel, dims) = all(dims .<= max_dims[1:length(dims)]) && prod(dims) <= KI.max_work_group_size(kernel)

        @testset "Sub-group layout" begin
            sub_group_layout_testsuite(backend, AT, sg_size, fits)
        end

        @testset "Sub-group return types" begin
            @test sg_size isa Int && sg_size >= 1

            types = filter(T -> KI.supports_shuffle(backend, T), [Int32, Float32, Int64, UInt32])
            if !isempty(types)
                results = KI.zeros(backend, Bool, 11)
                val = one(first(types))
                kernel = KI.@launch backend launch = false subgroup_typecheck_kernel(results, val)
                if fits(kernel, (sg_size,))
                    kernel(results, val; workgroupsize = sg_size)
                    KI.synchronize(backend)
                    @test all(Array(results))
                else
                    @test_skip "work-groups of $sg_size work-items"
                end
            end
        end

        # checks the sub-groups of a work-group of shape `dims`. Which work-items form a
        # sub-group, and how many sub-groups there are, is unspecified.
        function check_subgroups(data, dims)
            items = prod(dims)
            @test all(d -> d.max_sub_group_size == sg_size, data)
            # all work-items agree on the number of sub-groups, which is at least what full
            # sub-groups would need
            n = first(data).num_sub_groups
            @test all(d -> d.num_sub_groups == n, data)
            @test cld(items, sg_size) <= n <= items
            # the sub-group ids are 1:n
            @test sort(unique(map(d -> d.sub_group_id, data))) == 1:n
            # every sub-group has as many members as its size says, and they are its lanes
            for id in 1:n
                members = filter(d -> d.sub_group_id == id, data)
                size = length(members)
                @test 1 <= size <= sg_size
                @test all(d -> d.sub_group_size == size, members)
                @test sort(map(d -> d.sub_group_local_id, members)) == 1:size
            end
            # a 1-D work-group that fits a sub-group is one
            if length(dims) == 1 && items <= sg_size
                @test n == 1
            end
            return
        end

        # work-group shapes to check: 1-D ones around the width, and multi-dimensional ones
        # whose first dimension is or isn't a multiple of the width
        subgroup_shapes = unique(
            [
                (sg_size - 1,), (sg_size,), (sg_size + 1,), (2 * sg_size,), (2 * sg_size + 1,),
                (33, 2), (sg_size, 2), (sg_size + 1, 2), (7, 5), (5, 3, 2), (sg_size, 2, 2),
            ]
        )
        filter!(dims -> all(>(0), dims), subgroup_shapes)

        @testset "Sub-group formation" begin
            numgroups = 2
            @testset "$dims" for dims in subgroup_shapes
                items = prod(dims)
                results = AT(Vector{SubgroupData}(undef, items * numgroups))
                kernel = KI.@launch backend launch = false test_subgroup_kernel(results)
                if fits(kernel, dims)
                    kernel(results; workgroupsize = dims, numgroups)
                    KI.synchronize(backend)
                    for group in Iterators.partition(Array(results), items)
                        check_subgroups(collect(group), dims)
                    end
                else
                    @test_skip "work-groups of $dims work-items"
                end
            end
        end

        @testset "Combining a value per sub-group" begin
            numgroups = 3
            @testset "$dims" for dims in subgroup_shapes
                out = KI.zeros(backend, Int32, numgroups)
                kernel = KI.@launch backend launch = false subgroup_combine_kernel(out, Val(prod(dims)))
                if fits(kernel, dims)
                    kernel(out, Val(prod(dims)); workgroupsize = dims, numgroups)
                    KI.synchronize(backend)
                    @test all(==(prod(dims)), Array(out))
                else
                    @test_skip "work-groups of $dims work-items"
                end
            end
        end

        @testset "sub_group_barrier" begin
            out = KI.zeros(backend, Int32, sg_size, 2)
            scratch = KI.zeros(backend, Int32, sg_size)
            kernel = KI.@launch backend launch = false sub_group_barrier_kernel(scratch, out, Val(sg_size))
            if fits(kernel, (sg_size,))
                kernel(scratch, out, Val(sg_size); workgroupsize = sg_size)
                KI.synchronize(backend)
                other = mod1.(2:(sg_size + 1), sg_size)
                @test Array(out) == hcat(other, -other)
            else
                @test_skip "work-groups of $sg_size work-items"
            end
        end

        @testset "shuffles" begin
            candidates = (Int32, Int64, UInt32, Float32, Float64)
            types = filter(T -> KI.supports_shuffle(backend, T), candidates)
            @testset "$T" for T in types
                a = T.(rand(1:100, sg_size))
                @testset "shfl, shift $shift" for shift in unique((0, 1, sg_size - 1))
                    out = AT(zeros(T, sg_size))
                    KI.@launch backend workgroupsize = sg_size shfl_rotate_kernel(out, AT(a), shift)
                    KI.synchronize(backend)
                    @test Array(out) == circshift(a, -shift)
                end

                @testset "shfl_up, offset $offset" for offset in unique((1, 3, sg_size ÷ 2))
                    1 <= offset < sg_size || continue
                    out = AT(zeros(T, sg_size))
                    KI.@launch backend workgroupsize = sg_size shfl_up_lanes_kernel(out, T, offset)
                    KI.synchronize(backend)
                    @test Array(out)[(offset + 1):end] == T.(1:(sg_size - offset))
                    # lanes without a lane `offset` earlier get their own value
                    @test Array(out)[1:offset] == T.(1:offset)
                end

                if ispow2(sg_size)
                    out = AT(zeros(T, sg_size))
                    KI.@launch backend workgroupsize = sg_size shfl_xor_sum_kernel(out, AT(a), Val(sg_size))
                    KI.synchronize(backend)
                    @test all(==(sum(a)), Array(out))
                end
            end

            @testset "structs" begin
                T = ShuffleStruct
                @test KI.supports_shuffle(backend, T) ==
                    all(S -> KI.supports_shuffle(backend, S), (Float32, Int64, Int32))
                @test !KI.supports_shuffle(backend, Ref{Int})
                if KI.supports_shuffle(backend, T)
                    a = [T(i, -i, (i, 2i, 3i)) for i in 1:sg_size]
                    out = AT(fill(T(0, 0, (0, 0, 0)), sg_size))
                    KI.@launch backend workgroupsize = sg_size shfl_struct_kernel(out, AT(a))
                    KI.synchronize(backend)
                    @test Array(out) == circshift(a, -1)
                end
            end
        end

        subgroup_communication_testsuite(backend, AT, sg_size)

        @testset "votes" begin
            patterns = (
                "none" => falses(sg_size),
                "all" => trues(sg_size),
                "some" => [i % 3 == 1 for i in 1:sg_size],
                "last" => [i == sg_size for i in 1:sg_size],
            )
            @testset "$name" for (name, pred) in patterns
                out = AT(zeros(UInt64, sg_size, 4))
                KI.@launch backend workgroupsize = sg_size vote_kernel(out, AT(collect(pred)))
                KI.synchronize(backend)
                out = Array(out)
                @test all(==(any(pred)), out[:, 1])
                @test all(==(all(pred)), out[:, 2])
                @test all(==(sg_size), out[:, 4])
                if sg_size <= 64
                    mask = reduce(|, (UInt64(1) << (i - 1) for i in 1:sg_size if pred[i]); init = UInt64(0))
                    @test all(==(mask), out[:, 3])
                end
            end
        end

        @testset "shfl_down" begin
            candidates = (
                Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
                Float16, Float32, Float64,
            )
            types = filter(T -> KI.supports_shuffle(backend, T), candidates)
            @testset "$T" for T in types
                a = zeros(T, sg_size)
                rand!(a, (0:1))
                dev_a = AT(a)
                dev_b = AT(zeros(T, sg_size))
                KI.@launch backend workgroupsize = sg_size shfl_down_test_kernel(dev_a, dev_b, Val(sg_size))
                KI.synchronize(backend)
                @test sum(a) ≈ Array(dev_b)[1]

                # every lane whose source is in range gets the source's value
                @testset "offset $offset" for offset in unique((1, 3, sg_size ÷ 2))
                    1 <= offset < sg_size || continue
                    N = 2 * sg_size
                    out = KI.zeros(backend, T, N, 3)
                    kernel = KI.@launch backend launch = false shfl_down_lanes_kernel(out, T, offset)
                    # one sub-group if two don't fit in a work-group
                    fits(kernel, (N,)) || (N = sg_size)
                    kernel(out, T, offset; workgroupsize = N)
                    KI.synchronize(backend)
                    out = Array(out)
                    in_range = findall(i -> out[i, 1] + offset <= out[i, 2], 1:N)
                    # a work-group of one sub-group's width is a single sub-group; with more
                    # work-items, how many are in range depends on the sub-groups' sizes
                    if N == sg_size
                        @test length(in_range) == N - offset
                    end
                    @test !isempty(in_range)
                    @test out[in_range, 3] == out[in_range, 1] .+ offset
                    # a 1-D work-group forms full sub-groups (but the last one), and lanes
                    # past the sub-group width get their own value
                    past = findall(i -> out[i, 1] + offset > sg_size, 1:N)
                    @test !isempty(past)
                    @test out[past, 3] == out[past, 1]
                end
            end
        end
    end
    return nothing
end

# Checks that a backend implements the methods that have no fallback.
function contract_testsuite(backend::KI.Backend, AT)
    B = typeof(backend)
    @test hasmethod(KI.synchronize, Tuple{B})
    @test hasmethod(KI.copyto!, Tuple{B, AT, Array})
    @test hasmethod(KI.argconvert, Tuple{B, Any})
    @test hasmethod(KI.kernel_function, Tuple{B, Any, Type})
    @test hasmethod(KI.launch, Tuple{KI.Kernel{B}, Dims{3}, Dims{3}, Tuple})
    @test hasmethod(KI.max_work_group_size, Tuple{B})
    @test hasmethod(KI.max_work_group_size, Tuple{KI.Kernel{B}})
    @test hasmethod(KI.max_work_group_dims, Tuple{B})
    @test hasmethod(KI.max_num_groups, Tuple{B})
    if KI.supports_subgroups(backend)
        @test hasmethod(KI.sub_group_size, Tuple{B})
        # the device functions are overlays, so they can't be checked here; the sub-group
        # testsuite runs them
    end
    return
end
