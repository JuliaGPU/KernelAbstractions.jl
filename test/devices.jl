function devices_testsuite(Backend)
    backend = Backend()

    current_device = KernelAbstractions.device!(backend)
    for i in KernelAbstractions.ndevices(backend)
        KernelAbstractions.device!(backend, i)
        @test KernelAbstractions.device(backend) == i
    end

    @test_throws ArgumentError KernelAbstractions.device!(backend, 0)
    @test_throws ArgumentError KernelAbstractions.device!(backend, KernelAbstractions.ndevices(backend) + 1)
    return KernelAbstractions.device!(backend, current_device)
    KernelAbstractions.device!(backend, current_device)
end
