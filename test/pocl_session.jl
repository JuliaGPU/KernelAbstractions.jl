# Run by runtests.jl in a fresh process with several threads: it checks how the CPU back-end
# initializes its state, and how tasks on different threads share it.

using KernelAbstractions
using Test

import KernelAbstractions.POCL: POCL, @opencl

@kernel function fill_value!(A, v)
    I = @index(Global, Linear)
    @inbounds A[I] = v
end

@kernel function fill_rand!(A)
    I = @index(Global, Linear)
    @inbounds A[I] = rand(Float32)
end

@kernel function group_index!(G, L)
    I = @index(Global, Linear)
    @inbounds G[I] = @index(Group, Linear)
    @inbounds L[I] = @index(Local, Linear)
end

empty_kernel() = return

function pointer_value(out, p)
    @inbounds out[1] = UInt(reinterpret(Ptr{Cvoid}, p))
    return
end

# work-item i iterates n[i] times
function busy(A, n)
    i = POCL.get_global_id()
    x = 1.0
    for _ in 1:n[i]
        x = sin(x) + 1.0
    end
    A[i] = x
    return
end

@testset "POCL session" begin
    @test Threads.nthreads() > 1

    @testset "shared context, per-task queues" begin
        # concurrent first use initializes the back-end once
        sessions = fetch.([Threads.@spawn POCL.session() for _ in 1:16])
        @test all(s -> s === POCL.session(), sessions)

        queues = fetch.([Threads.@spawn (POCL.queue(), POCL.queue()) for _ in 1:16])
        @test all(((a, b),) -> a === b, queues)
        @test allunique(first.(queues))
    end

    @testset "kernels are linked once" begin
        kernels = fetch.([Threads.@spawn POCL.clfunction(empty_kernel).fun for _ in 1:16])
        @test all(k -> k === POCL.clfunction(empty_kernel).fun, kernels)
    end

    # kernel arguments are state of the kernel object, which all tasks share
    @testset "concurrent launches of one kernel" begin
        N = 4096
        arrays = [zeros(Float32, N) for _ in 1:32]
        @sync for (j, A) in enumerate(arrays)
            Threads.@spawn for _ in 1:10
                fill_value!(CPU())(A, Float32(j); ndrange = N)
            end
        end
        @test all(((j, A),) -> all(==(j), A), enumerate(arrays))

        # kernels using the RNG get extra arguments, sized by the workgroup size
        arrays = [fill(-1.0f0, N) for _ in 1:32]
        @sync for (j, A) in enumerate(arrays)
            Threads.@spawn for _ in 1:10
                fill_rand!(CPU(), isodd(j) ? 16 : 64)(A; ndrange = N)
            end
        end
        @test all(A -> all(x -> 0 <= x < 1, A), arrays)
    end

    # the default workgroup size spreads a launch over the threads
    @testset "default workgroup size" begin
        threads = KernelAbstractions.KernelInterface.multiprocessor_count(CPU())
        @test threads > 1
        N = 4096
        G = zeros(Int, N)
        L = zeros(Int, N)
        group_index!(CPU())(G, L; ndrange = N)
        # splitting stops with more than half as many workgroups as threads, or at the
        # minimum workgroup size of 32
        @test 2 * maximum(G) > min(threads, N ÷ 32)
        # every work-item ran once
        @test (G .- 1) .* maximum(L) .+ L == 1:N
    end

    @testset "null pointer arguments" begin
        out = UInt[1]
        x = Float32[0]
        P = Core.LLVMPtr{Float32, 1}
        function launch(p)
            GC.@preserve x @opencl pointer_value(out, p)
            return out[1]
        end
        @test launch(reinterpret(P, pointer(x))) == UInt(pointer(x))
        # the kernel mustn't see the pointer from the previous launch
        @test launch(reinterpret(P, C_NULL)) == 0
        # before Julia 1.12, a `Ptr` argument is an integer
        out[1] = 1
        @test launch(Ptr{Float32}(C_NULL)) == 0
    end

    # host kernel launches on several threads, one work-group per thread
    if POCL.device_limits().compute_units > 1
        nt = POCL.device_limits().compute_units
        sizes = (; global_size = (nt, 1, 1), local_size = (1, 1, 1))
        A = zeros(nt)
        n = zeros(Int, nt)
        k = POCL.clfunction(busy, POCL.argument_types((A, n)))
        h = POCL.host_kernel(k, sizes, nothing)
        @test h !== nothing
        # iterations per second
        k(A, n; sizes...)
        n[1] = 10^7
        rate = n[1] / @elapsed k(A, n; sizes...)

        # keep arguments alive until all threads finish, despite SIGINT (from a POSIX shell)
        # or cancellation on Julia 1.14+
        cancellation = isdefined(Base, :CANCEL_TOKEN)
        (cancellation || !Sys.iswindows()) &&
            @testset "interrupted launches ($path)" for path in (:short, :long)
            # one work-group takes 2 s, the others 0.5 s, and the interrupt comes after 0.25 s
            fill!(n, round(Int, rate / 2))
            n[end] = round(Int, 2rate)
            fill!(A, 0)
            # on a team of all threads
            Base.@atomic :monotonic h.last_duration = path == :short ? 0 : typemax(UInt64)
            POCL.record_serial_sample(h, nt, typemax(UInt64))
            POCL.record_serial_sample(h, nt, typemax(UInt64))
            @test POCL.host_schedule(h, POCL.session(), nt, path == :long)[1] == nt
            launch() = k(A, n; sizes...)
            t0 = time()
            err = try
                if cancellation
                    src = Base.CancellationTokenSource()
                    Threads.@spawn (sleep(0.25); Base.cancel!(src))
                    Base.ScopedValues.with(launch, Base.CANCEL_TOKEN => Base.CancellationToken(src))
                else
                    Base.exit_on_sigint(false)
                    run(`sh -c "sleep 0.25; kill -INT $(getpid())"`; wait = false)
                    launch()
                end
                nothing
            catch err
                err
            end
            elapsed = time() - t0
            # don't let a late interrupt hit later tests
            err === nothing && try
                sleep(5)
            catch
            end
            # defer the interrupt until the launch finishes
            @test elapsed > 0.5
            @test err isa (cancellation ? Base.CancellationRequest : InterruptException)
            @test all(!=(0), A)
        end

        # a single-threaded launch runs full and partial work-groups separately (here: two
        # handles of the same kernel); interrupts must wait until both runs finish
        (cancellation || !Sys.iswindows()) &&
            @testset "interrupted launch on one thread ($path)" for path in (:short, :long)
            s = POCL.session()
            full = @lock k.fun.lock POCL.create_host_kernel(k.fun, (1, 1, 1), false)
            B = zeros(2)
            m = [round(Int, rate / 2), 1]
            # one thread (short serial estimate), with either a short or long previous launch
            Base.@atomic :monotonic h.last_duration = path == :short ? 0 : typemax(UInt64)
            POCL.record_serial_sample(h, 2, UInt64(0))
            POCL.record_serial_sample(h, 2, UInt64(0))
            @test POCL.host_schedule(h, s, 2, path == :long) == (1, false)
            # the full work-group takes 0.5 s, and the interrupt comes after 0.25 s
            function launch()
                info = Ref(POCL.ExceptionInfo_st())
                GC.@preserve info begin
                    ptr = Base.unsafe_convert(Ptr{POCL.ExceptionInfo_st}, info)
                    POCL.host_launch_converted(k, POCL.converted_arguments(k, (B, m)), ptr, h, (2, 1, 1), full, (1, 1, 1))
                end
            end
            err = try
                if cancellation
                    src = Base.CancellationTokenSource()
                    Threads.@spawn (sleep(0.25); Base.cancel!(src))
                    Base.ScopedValues.with(launch, Base.CANCEL_TOKEN => Base.CancellationToken(src))
                else
                    Base.exit_on_sigint(false)
                    run(`sh -c "sleep 0.25; kill -INT $(getpid())"`; wait = false)
                    launch()
                end
                nothing
            catch err
                err
            end
            err === nothing && try
                sleep(5)
            catch
            end
            @test err isa (cancellation ? Base.CancellationRequest : InterruptException)
            # the partial work-group ran too
            @test B[2] != 0
        end

        @testset "garbage collection during a launch" begin
            # run one work-group for about a second while another thread collects garbage
            fill!(n, 0)
            n[1] = round(Int, rate)
            Base.@atomic :monotonic h.last_duration = 0
            t0 = time()
            gc = Threads.@spawn begin
                while time() - t0 < 0.2
                end
                @elapsed GC.gc()
            end
            k(A, n; global_size = (1, 1, 1), local_size = (1, 1, 1))
            elapsed = time() - t0
            # the collection doesn't wait for the kernel
            @test fetch(gc) < elapsed / 2
        end

        @testset "launch from a finalizer" begin
            # finalizers must run serially: waiting for a team may switch tasks
            fill!(n, round(Int, rate / 20))
            n[1] = 0
            fill!(A, 0)
            err = Ref{Any}(nothing)
            launched = Ref(false)
            # not inlined, so that nothing here keeps the object alive
            @noinline function garbage()
                finalizer(Ref(0)) do _
                    try
                        k(A, n; sizes...)
                    catch e
                        err[] = e
                    end
                    launched[] = true
                end
                return
            end
            garbage()
            # leave a task ready to expose an illegal switch from the finalizer
            k(zero(A), zero(n); sizes...)
            t = @async nothing
            GC.gc()
            wait(t)
            @test launched[]
            @test err[] === nothing
            @test all(!=(0), A)
        end
    end

    @testset "reset" begin
        s = POCL.session()
        q = POCL.queue()
        POCL.reset_session_state!()
        @test POCL.session() !== s
        @test POCL.queue() !== q
        A = zeros(Float32, 16)
        fill_value!(CPU())(A, 1.0f0; ndrange = 16)
        @test all(==(1), A)
    end
end
