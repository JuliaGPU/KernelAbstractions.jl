using KernelAbstractions
using KernelAbstractions.NDIteration
using InteractiveUtils
using LinearAlgebra
using SparseArrays
import SpecialFunctions

identity(x) = x

function unittest_testsuite(backend, backend_str, backend_mod, ArrayT, DeviceArrayT)
@testset "partition" begin
    let kernel = KernelAbstractions.Kernel{backend, StaticSize{(64,)}, DynamicSize, typeof(identity)}(identity)
        iterspace, dynamic = KernelAbstractions.partition(kernel, (128,), nothing)
        @test length(blocks(iterspace)) == 2
        @test dynamic isa NoDynamicCheck

        iterspace, dynamic = KernelAbstractions.partition(kernel, (129,), nothing)
        @test length(blocks(iterspace)) == 3
        @test dynamic isa DynamicCheck

        iterspace, dynamic = KernelAbstractions.partition(kernel, (129,), (64,))
        @test length(blocks(iterspace)) == 3
        @test dynamic isa DynamicCheck

        @test_throws ErrorException KernelAbstractions.partition(kernel, (129,), (65,))
    end
    let kernel = KernelAbstractions.Kernel{backend, StaticSize{(64,)}, StaticSize{(128,)}, typeof(identity)}(identity)
        iterspace, dynamic = KernelAbstractions.partition(kernel, (128,), nothing)
        @test length(blocks(iterspace)) == 2
        @test dynamic isa NoDynamicCheck

        iterspace, dynamic = KernelAbstractions.partition(kernel, nothing, nothing)
        @test length(blocks(iterspace)) == 2
        @test dynamic isa NoDynamicCheck

        @test_throws ErrorException KernelAbstractions.partition(kernel, (129,), nothing)
    end
end

@kernel function index_linear_global(A)
       I = @index(Global, Linear)
       A[I] = I
end
@kernel function index_linear_local(A)
       I  = @index(Global, Linear)
       i = @index(Local, Linear)
       A[I] = i
end
@kernel function index_linear_group(A)
       I  = @index(Global, Linear)
       i = @index(Group, Linear)
       A[I] = i
end
@kernel function index_cartesian_global(A)
       I = @index(Global, Cartesian)
       A[I] = I
end
@kernel function index_cartesian_local(A)
       I = @index(Global, Cartesian)
       i = @index(Local, Cartesian)
       A[I] = i
end
@kernel function index_cartesian_group(A)
       I = @index(Global, Cartesian)
       i = @index(Group, Cartesian)
       A[I] = i
end

@testset "get_device" begin
    x = ArrayT(rand(Float32, 5))
    A = ArrayT(rand(Float32, 5,5))
    device = backend()
    if isdefined(Main, :CUDA) && (device isa Main.CUDA.CUDAKernels.CUDADevice)
        deviceT = Main.CUDA.CUDAKernels.CUDADevice
    else
        deviceT = typeof(device)
    end
    @test @inferred(KernelAbstractions.get_device(A)) isa deviceT
    @test @inferred(KernelAbstractions.get_device(view(A, 2:4, 1:3))) isa deviceT
    if !(isdefined(Main, :ROCKernels) && (device isa Main.ROCKernels.ROCDevice)) &&
       !(isdefined(Main, :oneAPIKernels) && (device isa Main.oneAPIKernels.oneAPIDevice))
        # Sparse arrays are not supported by the ROCm or oneAPI backends yet:
        @test @inferred(KernelAbstractions.get_device(sparse(A))) isa deviceT
    end
    @test @inferred(KernelAbstractions.get_device(Diagonal(x))) isa deviceT
    @test @inferred(KernelAbstractions.get_device(Tridiagonal(A))) isa deviceT
end

@testset "indextest" begin
    # TODO: add test for _group and _local_cartesian
    A = ArrayT{Int}(undef, 16, 16)
    index_linear_global(backend(), 8)(A, ndrange=length(A))
    synchronize(backend())
    @test all(A .== LinearIndices(A))

    A = ArrayT{Int}(undef, 8)
    index_linear_local(backend(), 8)(A, ndrange=length(A))
    synchronize(backend())
    @test all(A .== 1:8)

    A = ArrayT{Int}(undef, 16)
    index_linear_local(backend(), 8)(A, ndrange=length(A))
    synchronize(backend())
    @test all(A[1:8] .== 1:8)
    @test all(A[9:16] .== 1:8)

    A = ArrayT{Int}(undef, 8, 2)
    index_linear_local(backend(), 8)(A, ndrange=length(A))
    synchronize(backend())
    @test all(A[1:8] .== 1:8)
    @test all(A[9:16] .== 1:8)

    A = ArrayT{CartesianIndex{2}}(undef, 16, 16)
    index_cartesian_global(backend(), 8)(A, ndrange=size(A))
    synchronize(backend())
    @test all(A .== CartesianIndices(A))

    A = ArrayT{CartesianIndex{1}}(undef, 16, 16)
    index_cartesian_global(backend(), 8)(A, ndrange=length(A))
    synchronize(backend())
    @test all(A[:] .== CartesianIndices((length(A),)))

    # Non-multiplies of the workgroupsize
    A = ArrayT{Int}(undef, 7, 7)
    index_linear_global(backend(), 8)(A, ndrange=length(A))
    synchronize(backend())
    @test all(A .== LinearIndices(A))

    A = ArrayT{Int}(undef, 5)
    index_linear_local(backend(), 8)(A, ndrange=length(A))
    synchronize(backend())
    @test all(A .== 1:5)
end

@kernel function constarg(A, @Const(B))
    I = @index(Global)
    @inbounds A[I] = B[I]
end

@testset "Const" begin
    let kernel = constarg(backend(), 8, (1024,))
        # this is poking at internals
        iterspace = NDRange{1, StaticSize{(128,)}, StaticSize{(8,)}}();
        ctx = if backend == CPU
            KernelAbstractions.mkcontext(kernel, 1, nothing, iterspace, Val(NoDynamicCheck()))
        else
            KernelAbstractions.mkcontext(kernel, nothing, iterspace)
        end
        AT = if backend == CPU
            Array{Float32, 2}
        else
            DeviceArrayT{Float32, 2, 1} # AS 1
        end
        IR = sprint() do io
            if backend_str == "CPU"
                code_llvm(io, kernel.f, (typeof(ctx), AT, AT),
                          optimize=false, raw=true)
            else
                backend_mod.code_llvm(io, kernel.f, (typeof(ctx), AT, AT),
                               kernel=true, optimize=true)
            end
        end
        if backend_str == "CPU"
            @test occursin("!alias.scope", IR)
            @test occursin("!noalias", IR)
        elseif backend_str == "CUDA"
            @test occursin("@llvm.nvvm.ldg", IR)
        elseif backend_str == "ROCM"
            @test occursin("addrspace(4)", IR)
        end
    end
end

@kernel function kernel_val!(a, ::Val{m}) where {m}
    I = @index(Global)
    @inbounds a[I] = m
end

A = convert(ArrayT, zeros(Int64, 1024))
kernel_val!(backend())(A,Val(3), ndrange=size(A))
synchronize(backend())
@test all((a)->a==3, A)

@kernel function kernel_empty()
    nothing
end

@testset "CPU synchronization" begin
    kernel_empty(CPU(), 1)(ndrange=1)
    synchronize(CPU())
end

@testset "Zero iteration space $backend" begin
    kernel_empty(backend(), 1)(ndrange=1)
    kernel_empty(backend(), 1)(ndrange=0)
    synchronize(backend())

    kernel_empty(backend(), 1)(ndrange=0)
    synchronize(backend())
end


@testset "return statement" begin
    try
        @eval @kernel function kernel_return()
            return
        end
        @test false
    catch e
        @test e.error ==
            ErrorException("Return statement not permitted in a kernel function kernel_return")
    end
end

@testset "fallback test: callable types" begin
    @eval begin
        function f end
        @kernel function (a::typeof(f))(x, ::Val{m}) where m
            I = @index(Global)
            @inbounds x[I] = m
        end
        @kernel function (a::typeof(f))(x, ::Val{1})
            I = @index(Global)
            @inbounds x[I] = 1
        end
        x = [1,2,3]
        env = f(CPU())(x, Val(4); ndrange=length(x))
        synchronize(CPU())
        @test x == [4,4,4]

        x = [1,2,3]
        env = f(CPU())(x, Val(1); ndrange=length(x))
        synchronize(CPU())
        @test x == [1,1,1]
    end
end

@testset "special functions: gamma" begin
    @eval begin
        @kernel function gamma_knl(A, @Const(B))
            I = @index(Global)
            @inbounds A[I] = SpecialFunctions.gamma(B[I])
        end

        x = Float32[1.0,2.0,3.0,5.5]
        y = similar(x)
        if $backend == CPU
            gamma_knl(CPU())(y, x; ndrange=length(x))
        else
            cx = $ArrayT(x)
            cy = similar(cx)
            gamma_knl($backend())(cy, cx; ndrange=length(x))
        end
        synchronize($backend())
        if $backend == CPU
            @test y == SpecialFunctions.gamma.(x)
        else
            cy = Array(cy)
            @test cy[1:4] ≈ SpecialFunctions.gamma.(x[1:4])
        end
    end
end

@testset "special functions: erf" begin
    @eval begin
        @kernel function erf_knl(A, @Const(B))
            I = @index(Global)
            @inbounds A[I] = SpecialFunctions.erf(B[I])
        end

        x = Float32[-1.0,-0.5,0.0,1e-3,1.0,2.0,5.5]
        y = similar(x)
        if $backend == CPU
            erf_knl(CPU())(y, x; ndrange=length(x))
        else
            cx = $ArrayT(x)
            cy = similar(cx)
            erf_knl($backend())(cy, cx; ndrange=length(x))
        end
        synchronize($backend())
        if $backend == CPU
            @test y == SpecialFunctions.erf.(x)
        else
            cy = Array(cy)
            @test cy[1:4] ≈ SpecialFunctions.erf.(x[1:4])
        end
    end
end

@testset "special functions: erfc" begin
    @eval begin
        @kernel function erfc_knl(A, @Const(B))
            I = @index(Global)
            @inbounds A[I] = SpecialFunctions.erfc(B[I])
        end

        x = Float32[-1.0,-0.5,0.0,1e-3,1.0,2.0,5.5]
        y = similar(x)
        if $backend == CPU
            erfc_knl(CPU())(y, x; ndrange=length(x))
        else
            cx = $ArrayT(x)
            cy = similar(cx)
            erfc_knl($backend())(cy, cx; ndrange=length(x))
        end
        synchronize($backend())
        if $backend == CPU
            @test y == SpecialFunctions.erfc.(x)
        else
            cy = Array(cy)
            @test cy[1:4] ≈ SpecialFunctions.erfc.(x[1:4])
        end
    end
end

end
