# INCLUDE ROCM
using KernelAbstractions, Test
using KernelAbstractions: @atomic, @atomicswap, @atomicreplace
include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend

# Function to use as a baseline for CPU metrics
function create_histogram(input)
    histogram_output = zeros(Int, maximum(input))
    for i in input
        histogram_output[i] += 1
    end
    return histogram_output
end

# This a 1D histogram kernel where the histogramming happens on shmem
@kernel function histogram_kernel!(histogram_output, input)
    tid = @index(Global, Linear)
    lid = @index(Local, Linear)

    @uniform warpsize = Int(32)

    @uniform gs = @groupsize()[1]
    @uniform N = length(histogram_output)

    shared_histogram = @localmem Int (gs)

    # This will go through all input elements and assign them to a location in
    # shmem. Note that if there is not enough shem, we create different shmem
    # blocks to write to. For example, if shmem is of size 256, but it's
    # possible to get a value of 312, then we will have 2 separate shmem blocks,
    # one from 1->256, and another from 256->512
    @uniform max_element = 1
    for min_element in 1:gs:N

        # Setting shared_histogram to 0
        @inbounds shared_histogram[lid] = 0
        @synchronize()

        max_element = min_element + gs
        if max_element > N
            max_element = N + 1
        end

        # Defining bin on shared memory and writing to it if possible
        bin = input[tid]
        if bin >= min_element && bin < max_element
            bin -= min_element - 1
            @atomic shared_histogram[bin] += 1
        end

        @synchronize()

        if ((lid + min_element - 1) <= N)
            @atomic histogram_output[lid + min_element - 1] += shared_histogram[lid]
        end

    end

end

function histogram!(histogram_output, input)
    backend = get_backend(histogram_output)
    # Need static block size
    kernel! = histogram_kernel!(backend, (256,))
    kernel!(histogram_output, input, ndrange = size(input))
    return
end

function move(backend, input)
    # TODO replace with adapt(backend, input)
    out = KernelAbstractions.allocate(backend, eltype(input), size(input))
    KernelAbstractions.copyto!(backend, out, input)
    return out
end

@testset "histogram tests" begin
    if Base.VERSION < v"1.7.0" && !KernelAbstractions.isgpu(backend)
        @test_skip false
    else
        rand_input = [rand(1:128) for i in 1:1000]
        linear_input = [i for i in 1:1024]
        all_two = [2 for i in 1:512]

        histogram_rand_baseline = create_histogram(rand_input)
        histogram_linear_baseline = create_histogram(linear_input)
        histogram_two_baseline = create_histogram(all_two)

        rand_input = move(backend, rand_input)
        linear_input = move(backend, linear_input)
        all_two = move(backend, all_two)

        rand_histogram = KernelAbstractions.zeros(backend, Int, 128)
        linear_histogram = KernelAbstractions.zeros(backend, Int, 1024)
        two_histogram = KernelAbstractions.zeros(backend, Int, 2)

        histogram!(rand_histogram, rand_input)
        histogram!(linear_histogram, linear_input)
        histogram!(two_histogram, all_two)
        KernelAbstractions.synchronize(CPU())

        @test isapprox(Array(rand_histogram), histogram_rand_baseline)
        @test isapprox(Array(linear_histogram), histogram_linear_baseline)
        @test isapprox(Array(two_histogram), histogram_two_baseline)
    end
end
