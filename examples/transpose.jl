using KernelAbstractions, CuArrays
using StaticArrays

@kernel function copy!(a,b)
    i = @index(Global)
    @inbounds b[i] = a[i]
end

@kernel function naive_transpose!(a,b, dim1, dim2)
    cartesian_in = CartesianIndices(a)[@index(Global)]

    # Changing index order for transpose.
    # Note: the only way I can see to do this is to transfer it into an array
    #       for a swap, and then transfer back
    tmp_idx = MVector(cartesian_in.I)
    @inbounds tmp_idx[dim1], tmp_idx[dim2] = tmp_idx[dim2], tmp_idx[dim1]
    cartesian_out = CartesianIndex(tmp_idx.data)

    b[cartesian_out] = a[cartesian_in]
end

function main()

    res = 8192
    # blocks must be defined based on tile size that can fit into shared mem
    tile_res = 1024
    #res = 1024

    a = round.(rand(Float32, (res, res))*100)
    #a = ones(Float32, (res, res))
    d_a = CuArray(a)
    d_b = CuArray(zeros(Float32, res, res))

    println("Copy time is:")
    CuArrays.@time copy!(CUDA(),1024)(d_a, d_b, ndrange=size(a))

    println("Transpose time is:")
    CuArrays.@time naive_transpose!(CUDA(),256)(d_a, d_b, 1, 2, ndrange=size(a))

    a = Array(d_a)
    b = Array(d_b)

    if (a == transpose(b))
    #if (a == b)
        println("Good job, man")
    else
        println("You failed. Sorry.")
        return a .- b
    end

    return nothing
end

main()

