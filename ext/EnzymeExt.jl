module EnzymeExt
    using EnzymeCore
    using EnzymeCore.EnzymeRules
    import KernelAbstractions: Kernel, wait

    #TODO: Keywords are the worst

    function EnzymeRules.augmented_primal(config::Config, func::Const{<:Kernel}, ::Duplicated{Event}, args...) where Event
        @show config
        @show func.val
        @show args
        @show Event

        # Pre-allocate tape according to ndrange... * inner_tape
        kernel = func.val
        f = kernel.f

        tt′ = Tuple{map(typeof, args)...}

        # TODO autodiff_deferred on the func.val
        # TODO modified between use config info
        forward, reverse = thunk(f, #=df=#nothing, Const{Nothing}, tt′,  Val(Enzyme.API.DEM_ReverseModePrimal), Val(get_width(config)), #=ModifiedBetween=#Val(true))
        TapeType = typeof(forward(ctx, args...))

        subtape = Vector{TapeType}(ndrange(...))

        function fwd(ctx, subtape, args...)
            subtape[ctx.idx] = forward(ctx, args...)
            return nothing
        end

        function rev(ctx, subtape, args...)
            reverse(ctx, args..., subtape[ctx.idx])
            return nothing
        end

        aug_kernel = similar(kernel)
        aug_kernel.f = fwd

        rev_kernel = similar(kernel)
        rev_kernel.f = rev

        primal = aug_kernel(subtape, args...)

        tape = Ref{Event}()
        function reverse_launch()
            tape[] = rev_kernel(subtape, args...)
        end
        shadow = reverse_launch
        return AugmentedReturnFlexShadow(primal, shadow, tape)
    end

    function reverse(::Config, ::Const{<:Kernel}, ::Type{<:Const}, tape, args...)
        wait(tape[])
        return ()
    end

    function EnzymeRules.augmented_primal(config::Config, func::Const{typeof(wait)}, RT, args...)
        primal = func.val(args...)
        if !EnzymeRules.needs_primal(config)
            primal = nothing
        end
        return AugmentedReturn(primal, nothing, nothing)
    end

    function reverse(::Config, ::Const{typeof(wait)}, ::Type{<:Const}, tape, arg)
        # This duplicated is invalid, it contains the fake shadow
        closure = arg.dval
        ev = closure()

        wait(rev)

        return ()
    end
end
