using KernelAbstractions, Test, CUDA

if has_cuda_gpu()
    CUDA.allowscalar(false)
end

@testset "Error propagation" begin
    let event = Event(()->error(""))
        @test_throws TaskFailedException wait(event)
    end

    let event = Event(error, "")
        @test_throws CompositeException wait(MultiEvent(event))
    end

    let event = Event(error, "")
        event = Event(wait, MultiEvent(event))
        @test_throws TaskFailedException wait(event)
    end

    let event = Event(error, "")
        event = Event(()->nothing, dependencies=event)
        @test_throws TaskFailedException wait(event)
    end
end

if has_cuda_gpu()
    barrier = Base.Threads.Event()
    cpu_event = Event(wait, barrier)

    KernelAbstractions.unsafe_wait(CUDAGPU(), cpu_event) # Event edge on CuDefaultStream
    gpu_event = Event(CUDAGPU()) # Event on CuDefaultStream

    notify(barrier)
    wait(gpu_event)
end
