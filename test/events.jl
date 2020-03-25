using KernelAbstractions, Test, CUDAapi
if has_cuda_gpu()
    using CuArrays, CUDAdrv
    CuArrays.allowscalar(false)
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

    wait(CUDA(), cpu_event) # Event edge on CuDefaultStream
    gpu_event = Event(CUDA()) # Event on CuDefaultStream
    notify(barrier)
    wait(gpu_event)

    @kernel function happy()
        @print("I am so happy")
    end
    let kernel = happy(CUDA(), (1,))
        # precompile
        gpu_event = kernel(;ndrange=(1,))
        wait(gpu_event)

        barrier = Base.Threads.Event()
        cpu_event = Event(wait, barrier)
        notify(barrier)
        gpu_event = kernel(;ndrange=(1,), dependencies=cpu_event)
        wait(gpu_event)

        barrier = Base.Threads.Event()
        cpu_event = Event(wait, barrier)
        gpu_event = kernel(;ndrange=(1,), dependencies=cpu_event)
        notify(barrier)
        wait(gpu_event)
    end
    @kernel function happy()
        @print("I am so happy")
    end
    # let kernel = happy(CUDA(), (1,))
    #     # do not precompile, this hangs since cuModuleLoadDataEx will block
    #     # if device is busy
    #     barrier = Base.Threads.Event()
    #     cpu_event = Event(wait, barrier)
    #     gpu_event = kernel(;ndrange=(1,), dependencies=cpu_event)
    #     notify(barrier)
    #     wait(gpu_event)
    # end
end