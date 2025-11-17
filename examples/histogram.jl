# INCLUDE ROCM
using KernelAbstractions, Test
using KernelAbstractions: @atomic, @atomicswap, @atomicreplace
import KernelAbstractions.KernelIntrinsics as KI

include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend

# Function to use as a baseline for CPU metrics
function create_histogram(input)
    histogram_output = zeros(eltype(input), maximum(input))
    for i in input
        histogram_output[i] += 1
    end
    return histogram_output
end

# This a 1D histogram kernel where the histogramming happens on static shmem
function histogram_kernel!(histogram_output, input, ::Val{gs}) where {gs}
    gid = KI.get_group_id().x
    lid = KI.get_local_id().x

    tid = (gid - 1) * gs + lid
    N = length(histogram_output)

    shared_histogram = KI.localmemory(eltype(input), gs)

    # This will go through all input elements and assign them to a location in
    # shmem. Note that if there is not enough shem, we create different shmem
    # blocks to write to. For example, if shmem is of size 256, but it's
    # possible to get a value of 312, then we will have 2 separate shmem blocks,
    # one from 1->256, and another from 256->512
    for min_element in 1:gs:N

        # Setting shared_histogram to 0
        @inbounds shared_histogram[lid] = 0
        KI.barrier()

        max_element = min_element + gs
        if max_element > N
            max_element = N + 1
        end

        # Defining bin on shared memory and writing to it if possible
        bin = tid <= length(input) ? input[tid] : 0
        if bin >= min_element && bin < max_element
            bin -= min_element - 1
            @atomic shared_histogram[bin] += 1
        end

        KI.barrier()

        if ((lid + min_element - 1) <= N)
            @atomic histogram_output[lid + min_element - 1] += shared_histogram[lid]
        end

    end
    return
end

function histogram!(histogram_output, input, groupsize = 256)
    backend = get_backend(histogram_output)
    # Need static block size
    KI.@kernel backend workgroupsize = groupsize numworkgroups = cld(length(input), groupsize) histogram_kernel!(histogram_output, input, Val(groupsize))
    return
end

function move(backend, input)
    # TODO replace with adapt(backend, input)
    out = KernelAbstractions.allocate(backend, eltype(input), size(input))
    KernelAbstractions.copyto!(backend, out, input)
    return out
end

@testset "histogram tests" begin
    # Use Int32 as some backends don't support 64-bit atomics
    rand_input = Int32.(rand(1:128, 1000))
    linear_input = Int32.(1:1024)
    all_two = fill(Int32(2), 512)

    histogram_rand_baseline = create_histogram(rand_input)
    histogram_linear_baseline = create_histogram(linear_input)
    histogram_two_baseline = create_histogram(all_two)

    rand_input = move(backend, rand_input)
    linear_input = move(backend, linear_input)
    all_two = move(backend, all_two)

    rand_histogram = KernelAbstractions.zeros(backend, eltype(rand_input), Int(maximum(rand_input)))
    linear_histogram = KernelAbstractions.zeros(backend, eltype(linear_input), Int(maximum(linear_input)))
    two_histogram = KernelAbstractions.zeros(backend, eltype(all_two), Int(maximum(all_two)))

    histogram!(rand_histogram, rand_input, 6)
    histogram!(linear_histogram, linear_input)
    histogram!(two_histogram, all_two)
    KernelAbstractions.synchronize(backend)

    @test isapprox(Array(rand_histogram), histogram_rand_baseline)
    @test isapprox(Array(linear_histogram), histogram_linear_baseline)
    @test isapprox(Array(two_histogram), histogram_two_baseline)
end
