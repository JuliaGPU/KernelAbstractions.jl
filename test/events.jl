using KernelAbstractions
using Test

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
