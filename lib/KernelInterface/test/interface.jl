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
    if KI.get_local_id().x == 1
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

# The tests of the communication functions below don't assume which work-items form a
# sub-group (unless they are about `supports_linear_subgroups`): the kernels run in 1-D
# work-groups, every work-item records its sub-group, lane and sub-group size at its local
# index, and the expected results are computed on the host for the sub-groups that formed.

@inline function record_sub_group!(ids, i)
    @inbounds begin
        ids[1, i] = KI.get_sub_group_id()
        ids[2, i] = KI.get_sub_group_local_id()
        ids[3, i] = KI.get_sub_group_size()
    end
    return
end

# every work-item applies `f`, which communicates within the sub-group, to its value of `a`
function sub_group_apply_kernel(ids, out, a, f)
    i = KI.get_local_id().x
    record_sub_group!(ids, i)
    @inbounds out[i] = f(a[i])
    return
end

# the work-items of every sub-group by lane, from the ids `record_sub_group!` wrote
function observed_sub_groups(ids)
    groups = [Int[] for _ in 1:maximum(ids[1, :])]
    for i in axes(ids, 2)
        push!(groups[ids[1, i]], i)
    end
    for g in groups
        sort!(g; by = i -> ids[2, i])
        @test [ids[2, i] for i in g] == 1:length(g)
        @test all(i -> ids[3, i] == length(g), g)
    end
    return groups
end

# Run `f` on the values `a` in a 1-D work-group of `length(a)` work-items, with results of
# type `T`. Returns the results by local index and the work-items of each sub-group, or
# `nothing` if the work-group is too large for the kernel.
function sub_group_apply(backend, AT, f, a, T = eltype(a))
    n = length(a)
    ids = AT(zeros(Int, 3, n))
    out = AT(Vector{T}(undef, n))
    dev_a = AT(a)
    kernel = KI.@launch backend launch = false sub_group_apply_kernel(ids, out, dev_a, f)
    n <= KI.max_work_group_size(kernel) || return nothing
    kernel(ids, out, dev_a, f; workgroupsize = n)
    KI.synchronize(backend)
    return Array(out), observed_sub_groups(Array(ids))
end

# work-group sizes that form full and partial sub-groups, several of them, and a single
# work-item
sub_group_test_sizes(sg_size) = unique((sg_size, max(sg_size - 3, 1), 2 * sg_size + 5, 1))

# The shuffles, as callables that are the same type for every offset, so that a kernel
# is compiled once for all of them.
struct Shuffle{F, A}
    f::F
    arg::A
end
(s::Shuffle)(x) = s.f(x, s.arg)

struct WidthShuffle{F, A}
    f::F
    arg::A
    width::Int
end
(s::WidthShuffle)(x) = s.f(x, s.arg, s.width)

# read from the lane `shift` further, wrapping around within the sub-group
struct Rotate
    shift::Int
end
(r::Rotate)(x) = KI.shfl(x, mod1(KI.get_sub_group_local_id() + r.shift, KI.get_sub_group_size()))

# apply `f` to values of type `T`, stored as values of another type of the same size, so
# that types the device can't compute with (e.g. `Float64`) are only moved
struct AsType{T, F}
    f::F
end
AsType{T}(f) where {T} = AsType{T, typeof(f)}(f)
(s::AsType{T})(x) where {T} = reinterpret(typeof(x), s.f(reinterpret(T, x)))

# The lane a shuffle reads from, given the lane and size of the sub-group of the work-item
# and the width `W`, or `nothing` where the result is unspecified.
in_sub_group(src, size) = src <= size ? src : nothing
function shuffle_source(s::Shuffle, l, size, W)
    arg = s.arg
    src = s.f === KI.shfl ? arg :
        s.f === KI.shfl_down ? (arg > W - l ? l : l + arg) :
        s.f === KI.shfl_up ? (l > arg ? l - arg : l) :
        ((l - 1) ⊻ arg) + 1
    return in_sub_group(src, size)
end
function shuffle_source(s::WidthShuffle, l, size, W)
    arg, w = s.arg, s.width
    pos = (l - 1) % w
    src = s.f === KI.shfl ? l - 1 - pos + mod1(arg, w) :
        s.f === KI.shfl_down ? (arg < w - pos ? l + arg : l) :
        s.f === KI.shfl_up ? (pos >= arg ? l - arg : l) :
        ((l - 1) ⊻ arg) + 1
    return in_sub_group(src, size)
end
shuffle_source(r::Rotate, l, size, W) = mod1(l + r.shift, size)
shuffle_source(s::AsType, l, size, W) = shuffle_source(s.f, l, size, W)

function shuffle_testsuite(backend, AT, sg_size, f, a)
    res = sub_group_apply(backend, AT, f, a)
    res === nothing && return
    out, groups = res
    ok = true
    for g in groups, (l, i) in enumerate(g)
        src = shuffle_source(f, l, length(g), sg_size)
        src === nothing && continue
        ok &= out[i] === a[g[src]]
    end
    @test ok
    return
end

# the shuffles that are tested for every supported type, `wrap` turning them into the
# shuffle that is run
function basic_shuffles(sg_size)
    return [
        Rotate(1), Rotate(sg_size - 1), Shuffle(KI.shfl, 1),
        Shuffle(KI.shfl_down, 1), Shuffle(KI.shfl_up, 1), Shuffle(KI.shfl_xor, 1),
    ]
end

function all_shuffles(sg_size)
    shuffles = Any[Rotate(0), Rotate(1), Rotate(sg_size - 1)]
    for lane in unique((1, 2, sg_size, sg_size + 1))
        push!(shuffles, Shuffle(KI.shfl, lane))
    end
    # offsets past the width, also ones that don't fit in 32 bits
    for d in unique((0, 1, 3, sg_size ÷ 2, sg_size - 1, sg_size, sg_size + 1, Int64(2)^32 + 1, typemax(Int64)))
        push!(shuffles, Shuffle(KI.shfl_down, d), Shuffle(KI.shfl_up, d))
    end
    for mask in unique((0, 1, 3, sg_size ÷ 2, sg_size - 1))
        0 <= mask < sg_size && push!(shuffles, Shuffle(KI.shfl_xor, mask))
    end
    for w in (1, 2, 4, 8, 16, 32, 64)
        w <= sg_size && ispow2(sg_size) || continue
        for arg in unique((1, 3, w - 1, w + 2, Int64(2)^32 + 1, typemax(Int64)))
            push!(shuffles, WidthShuffle(KI.shfl, arg, w))
            push!(shuffles, WidthShuffle(KI.shfl_down, arg, w))
            push!(shuffles, WidthShuffle(KI.shfl_up, arg, w))
        end
        for mask in unique((0, 1, w - 1))
            0 <= mask < w && push!(shuffles, WidthShuffle(KI.shfl_xor, mask, w))
        end
    end
    return shuffles
end

# The collectives, as callables
struct Reduce{O}
    op::O
end
(r::Reduce)(x) = KI.sub_group_reduce(r.op, x)

struct Scan{O}
    op::O
end
(s::Scan)(x) = KI.sub_group_scan(s.op, x)

struct ExclusiveScan{O, T}
    op::O
    init::T
end
(s::ExclusiveScan)(x) = KI.sub_group_exclusive_scan(s.op, x, s.init)

# an associative but not commutative operator: the composition of affine maps
compose_affine(f, g) = (g[1] * f[1], g[1] * f[2] + g[2])
# the (value, index) of the smallest value, the first one of equal values
argmin_op(x, y) = ifelse(y[1] < x[1], y, x)

# The expected result of a collective for lane `l` of the sub-group with the values `vals`
collective_result(r::Reduce, vals, l) = foldl(r.op, vals)
collective_result(s::Scan, vals, l) = foldl(s.op, vals[1:l])
collective_result(s::ExclusiveScan, vals, l) = l == 1 ? s.init : s.op(s.init, foldl(s.op, vals[1:(l - 1)]))

# Floating-point `+` and `*` may be reassociated, so they are compared with a tolerance
# (other than for the `init` of an exclusive scan, which lane 1 gets unchanged).
collective_op(f) = f.op
reassociable(f, ::Type{T}) where {T} = T <: AbstractFloat && collective_op(f) in (+, *)

function collective_testsuite(backend, AT, f, a)
    res = sub_group_apply(backend, AT, f, a)
    res === nothing && return
    out, groups = res
    T = eltype(a)
    ok = true
    for g in groups
        vals = a[g]
        for (l, i) in enumerate(g)
            expected = collective_result(f, vals, l)
            if reassociable(f, T) && !(f isa ExclusiveScan && l == 1)
                ok &= isapprox(out[i], expected; nans = true, rtol = sqrt(eps(T)))
            else
                # `isequal`, as the results may be NaN or a signed zero
                ok &= isequal(out[i], expected)
            end
        end
    end
    @test ok
    return
end

function vote_testsuite(backend, AT, sg_size, pred)
    for (vote, T) in ((KI.sub_group_any, Bool), (KI.sub_group_all, Bool), (KI.sub_group_ballot, UInt64))
        vote === KI.sub_group_ballot && sg_size > 64 && continue
        res = sub_group_apply(backend, AT, vote, pred, T)
        res === nothing && continue
        out, groups = res
        ok = true
        for g in groups
            p = pred[g]
            expected = vote === KI.sub_group_any ? any(p) :
                vote === KI.sub_group_all ? all(p) :
                reduce(|, (UInt64(1) << (l - 1) for l in eachindex(p) if p[l]); init = UInt64(0))
            ok &= all(i -> out[i] == expected, g)
        end
        @test ok
    end
    return
end

# the values come from a divergent branch, like for the padding work-items of a `@kernel`
function reduce_divergent_kernel(ids, red, scan, a, m)
    i = KI.get_local_id().x
    record_sub_group!(ids, i)
    val = i <= m ? (@inbounds a[i]) : zero(eltype(a))
    r = KI.sub_group_reduce(+, val)
    s = KI.sub_group_scan(+, val)
    @inbounds red[i] = r
    @inbounds scan[i] = s
    return
end

function reduce_divergent_testsuite(backend, AT, sg_size, ::Type{T}) where {T}
    n = sg_size
    m = max(n - 5, 1)
    a = T.(rand(1:20, n))
    ids = AT(zeros(Int, 3, n))
    red = AT(zeros(T, n))
    scan = AT(zeros(T, n))
    KI.@launch backend workgroupsize = n reduce_divergent_kernel(ids, red, scan, AT(a), m)
    KI.synchronize(backend)
    red, scan = Array(red), Array(scan)
    vals = [i <= m ? a[i] : zero(T) for i in 1:n]
    for g in observed_sub_groups(Array(ids))
        @test all(i -> red[i] == sum(vals[g]), g)
        @test all(l -> scan[g[l]] == sum(vals[g[1:l]]), eachindex(g))
    end
    return
end

# Every sub-group executes the communication functions a different number of times, and
# some return early, which needs `supports_independent_subgroups`.
function independent_sub_groups_kernel(ids, out, a)
    i = KI.get_local_id().x
    record_sub_group!(ids, i)
    sg = KI.get_sub_group_id()
    next = mod1(KI.get_sub_group_local_id() + 1, KI.get_sub_group_size())
    val = @inbounds a[i]
    acc = zero(val)
    for _ in 1:sg
        acc += KI.shfl(val, next)
    end
    @inbounds out[i] = acc
    iseven(sg) && return
    KI.sub_group_barrier()
    if KI.sub_group_any(val > Int32(1000))
        acc = -acc
    end
    @inbounds out[i] = acc + KI.shfl(val, 1)
    return
end

function independent_sub_groups_testsuite(backend, AT, sg_size)
    n = 4 * sg_size
    a = Int32.(rand(1:2000, n))
    ids, out = AT(zeros(Int, 3, n)), AT(zeros(Int32, n))
    kernel = KI.@launch backend launch = false independent_sub_groups_kernel(ids, out, AT(a))
    n = min(n, KI.max_work_group_size(kernel))
    kernel(ids, out, AT(a); workgroupsize = n)
    KI.synchronize(backend)
    out = Array(out)
    groups = observed_sub_groups(Array(ids)[:, 1:n])
    for (sg, g) in enumerate(groups), (l, i) in enumerate(g)
        expected = Int32(sg) * a[g[mod1(l + 1, length(g))]]
        if isodd(sg)
            any(>(1000), a[g]) && (expected = -expected)
            expected += a[g[1]]
        end
        @test out[i] == expected
    end
    return
end

# Every work-item writes local and global memory at its sub-group slot, and reads the next
# lane's write after a sub-group barrier. `N` bounds the slots: the work-group size times the
# width.
function sub_group_barrier_kernel(ids, scratch, out, ::Val{N}) where {N}
    i = KI.get_local_id().x
    record_sub_group!(ids, i)
    lane = KI.get_sub_group_local_id()
    base = (KI.get_sub_group_id() - 1) * KI.get_max_sub_group_size()
    other = base + mod1(lane + 1, KI.get_sub_group_size())
    lm = KI.localmemory(Int32, N)
    @inbounds lm[base + lane] = i
    @inbounds scratch[base + lane] = -i
    KI.sub_group_barrier()
    @inbounds out[i, 1] = lm[other]
    @inbounds out[i, 2] = scratch[other]
    return
end

struct ShuffleStruct
    a::Float32
    b::Int64
    c::NTuple{3, Int32}
end

# primitive types that no backend supports natively, shuffled as `UInt32` words
primitive type Bits64 64 end
primitive type Bits16 16 end
Bits64(x::Integer) = reinterpret(Bits64, x % UInt64)
Bits16(x::Integer) = reinterpret(Bits16, x % UInt16)

struct FallbackStruct
    flag::Bool
    c::Char
    x::Bits64
    limbs::NTuple{4, Int64}
end

# a kernel whose callable captures an array, compiled but not launched yet
function captured_array_kernel(backend, AT, out)
    a = AT(Int32[42])
    kernel = KI.@launch backend launch = false (() -> (@inbounds out[1] = a[1]; nothing))()
    return kernel, WeakRef(a)
end

# 1-D work-groups and ones whose x extent is a multiple of the sub-group width form
# sub-groups from consecutive work-items, x fastest
function sub_group_layout_testsuite(backend, AT, sg_size, fits)
    # including one with a partial last sub-group past 256 work-items, which catches 8-bit
    # arithmetic in the index computations
    shapes = (
        (sg_size + 5,), (3 * sg_size,), (9 * sg_size + 5,),
        (sg_size, 4), (2 * sg_size, 2), (sg_size, 2, 2),
    )
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

# The sub-group communication functions. A separate function, so that `interface_testsuite`
# doesn't get too large to compile.
function subgroup_communication_testsuite(backend::KI.Backend, AT, sg_size)
    sizes = sub_group_test_sizes(sg_size)
    fp64 = KI.supports_float64(backend)

    @testset "sub_group_barrier" begin
        n = sg_size
        ids = AT(zeros(Int, 3, n))
        scratch = AT(zeros(Int32, n * sg_size))
        out = AT(zeros(Int32, n, 2))
        KI.@launch backend workgroupsize = n sub_group_barrier_kernel(ids, scratch, out, Val(n * sg_size))
        KI.synchronize(backend)
        out = Array(out)
        for g in observed_sub_groups(Array(ids)), (l, i) in enumerate(g)
            other = g[mod1(l + 1, length(g))]
            @test out[i, :] == [other, -other]
        end
    end

    @testset "votes" begin
        @testset "$n work-items" for n in sizes
            for pred in (falses(n), trues(n), [i % 3 == 1 for i in 1:n], [i == n for i in 1:n], rand(Bool, n))
                vote_testsuite(backend, AT, sg_size, collect(pred))
            end
        end
    end

    KI.supports_shuffle(backend, UInt32) || return

    @testset "shuffles" begin
        @testset "$n work-items" for n in sizes
            a = Int32.(1:n) .* Int32(10)
            for f in all_shuffles(sg_size)
                shuffle_testsuite(backend, AT, sg_size, f, a)
            end
        end

        # other types, natively or as words or fields
        values = Any[
            Int8 => i -> Int8(i % 100), UInt8 => i -> UInt8(i), Int16 => i -> Int16(-i),
            UInt16 => i -> (1000i) % UInt16, UInt32 => i -> UInt32(i) << 20, Float16 => Float16,
            Int64 => i -> (Int64(i) << 40) - i, UInt64 => i -> typemax(UInt64) - i,
            Float32 => i -> Float32(i) / 3, Bool => isodd, Char => i -> Char(0x0001F600 + i),
            Bits64 => i -> Bits64((UInt64(i) << 40) - i), Bits16 => i -> Bits16(0xa000 + i),
            ShuffleStruct => i -> ShuffleStruct(i, -i, (i, 2i, 3i)),
            FallbackStruct => i -> FallbackStruct(isodd(i), Char(64 + i), Bits64(-i), (i, -i, 2i, typemin(Int64) + i)),
        ]
        fp64 && push!(values, Float64 => i -> 1 / i)
        @testset "$T" for (T, f) in values
            KI.supports_shuffle(backend, T) || continue
            @testset "$n work-items" for n in sizes
                a = T[f(i) for i in 1:n]
                for s in basic_shuffles(sg_size)
                    shuffle_testsuite(backend, AT, sg_size, s, a)
                end
            end
        end

        # `Float64` values are moved through `UInt64` storage, so that this also works where
        # the device can't compute with them
        @testset "Float64 as UInt64" begin
            if KI.supports_shuffle(backend, Float64)
                a = [reinterpret(UInt64, 1 / i) for i in 1:sg_size]
                for s in basic_shuffles(sg_size)
                    shuffle_testsuite(backend, AT, sg_size, AsType{Float64}(s), a)
                end
            end
        end

        # word and field fallbacks are derived from `UInt32`
        @test KI.supports_shuffle(backend, Bool)
        @test KI.supports_shuffle(backend, Char)
        @test KI.supports_shuffle(backend, Bits64)
        @test KI.supports_shuffle(backend, Bits16)
        @test KI.supports_shuffle(backend, FallbackStruct)
        @test KI.supports_shuffle(backend, ShuffleStruct) ==
            all(S -> KI.supports_shuffle(backend, S), (Float32, Int64, Int32))
        @test !KI.supports_shuffle(backend, Ref{Int})
    end

    @testset "sub_group_reduce and scans" begin
        @testset "$n work-items" for n in sizes
            ops = Any[
                (+, Int32.(rand(1:100, n))),
                # wrap-around
                (+, Int32[typemax(Int32) - rand(Int32(0):Int32(10)) for _ in 1:n]),
                (*, Int32.(rand(-3:3, n))),
                # operators and types that backends may implement natively
                (+, UInt64.(rand(1:100, n))), (+, Int64.(rand(-100:100, n))),
                (+, UInt32.(rand(1:100, n))),
                (min, Int64.(rand(-100:100, n))), (min, Int32.(rand(-100:100, n))),
                (max, UInt32.(rand(1:100, n))), (max, Int64.(rand(-100:100, n))),
                (|, UInt32.(rand(UInt32, n))), (&, UInt32.(rand(UInt32, n))),
                (xor, UInt64.(rand(UInt64, n))),
                # not commutative
                (compose_affine, [(Int32(rand((-1, 1, 2))), Int32(rand(-5:5))) for _ in 1:n]),
                (argmin_op, [(Float32(rand(1:20)), Int32(i)) for i in 1:n]),
            ]
            float_types = Any[Float32]
            KI.supports_shuffle(backend, Float16) && push!(float_types, Float16)
            fp64 && KI.supports_shuffle(backend, Float64) && push!(float_types, Float64)
            for T in float_types
                # exactly representable sums for `Float16`, rounding for the others
                push!(ops, (+, T === Float16 ? T.(rand(1:8, n)) ./ 4 : T.(rand(n))))
                T === Float16 || push!(ops, (*, T.(rand(0.9:0.01:1.1, n))))
                # NaN and infinities propagate through `+` whatever the order
                push!(ops, (+, T[i == 2 ? -T(Inf) : i == 3 ? T(Inf) : rand(1:100) for i in 1:n]))
                push!(ops, (+, T[i == 2 ? T(NaN) : rand(1:100) for i in 1:n]))
                # Julia's `max` and `min` propagate NaN, unlike OpenCL's
                push!(ops, (max, T[i == 2 ? T(NaN) : rand(1:100) for i in 1:n]))
                push!(ops, (min, T[i == n ? T(NaN) : rand(1:100) for i in 1:n]))
                # ... and tell the sign of zero apart
                push!(ops, (min, T[isodd(i) ? -zero(T) : zero(T) for i in 1:n]))
                push!(ops, (max, T[isodd(i) ? zero(T) : -zero(T) for i in 1:n]))
            end
            for (op, a) in ops
                KI.supports_shuffle(backend, eltype(a)) || continue
                collective_testsuite(backend, AT, Reduce(op), a)
                collective_testsuite(backend, AT, Scan(op), a)
            end

            # exclusive scans with an `init` that isn't the identity
            collective_testsuite(backend, AT, ExclusiveScan(+, Int32(7)), Int32.(rand(1:100, n)))
            collective_testsuite(backend, AT, ExclusiveScan(+, UInt64(7)), UInt64.(rand(1:100, n)))
            collective_testsuite(backend, AT, ExclusiveScan(max, Int32(50)), Int32.(rand(1:100, n)))
            collective_testsuite(backend, AT, ExclusiveScan(+, 7.0f0), Float32.(rand(1:100, n)))
            collective_testsuite(
                backend, AT, ExclusiveScan(compose_affine, (Int32(2), Int32(3))),
                [(Int32(rand((-1, 1, 2))), Int32(rand(-5:5))) for _ in 1:n]
            )
            # lane 1 gets `init` itself, not `init + 0.0`
            collective_testsuite(backend, AT, ExclusiveScan(+, -0.0f0), Float32.(rand(1:100, n)))
            collective_testsuite(backend, AT, ExclusiveScan(+, -0.0f0), fill(-0.0f0, n))
        end
    end

    @testset "sub_group_reduce and sub_group_scan of divergent values" begin
        reduce_divergent_testsuite(backend, AT, sg_size, Int32)
        reduce_divergent_testsuite(backend, AT, sg_size, Float32)
    end

    if KI.supports_independent_subgroups(backend)
        @testset "independent sub-groups" begin
            independent_sub_groups_testsuite(backend, AT, sg_size)
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
        @test KI.supports_linear_subgroups(b) isa Bool
        @test KI.supports_independent_subgroups(b) isa Bool
        # both imply sub-group support
        KI.supports_linear_subgroups(b) && @test KI.supports_subgroups(b)
        KI.supports_independent_subgroups(b) && @test KI.supports_subgroups(b)
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
            if KI.supports_linear_subgroups(backend)
                sub_group_layout_testsuite(backend, AT, sg_size, fits)
            end
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
            # with the linear layout, a 1-D work-group that fits a sub-group is one
            if KI.supports_linear_subgroups(backend) && length(dims) == 1 && items <= sg_size
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

        subgroup_communication_testsuite(backend, AT, sg_size)
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
