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
        # kernel = func.val
        # f = kernel.f
        # function df(ctx, args...)
        #     Enzyme.autodiff_deferred(Split, f, Enzyme.Const, ctx, args...)
        #     return nothing
        # end
        # kernel' = similar(kernel, df)

        # TODO autodiff_deferred on the func.val
        forward, reverse = thunk(...)

        subtape = nothing
        if EnzymeRules.needs_primal(config)
            primal, subtape = forward(args...)
        else
            primal = nothing
        end

        tape = Ref{Event}()
        if EnzymeRules.needs_shadow(config)
            function reverse_launch()
                tape[] = reverse(args..., subtape)
            end
            shadow = reverse_launch
        else
            shadow = nothing
        end
        return AugmentedReturnFlexShadow(primal, shadow, tape)
    end

    function reverse(::Config, ::Const{<:Kernel}, ::Type{<:Const}, tape, args...)
        wait(tape[])
        return ()
    end

    function EnzymeRules.augmented_primal(config::Config, func::Const{typeof(wait)}, RT, args...)
        if EnzymeRules.needs_primal(config)
            primal = func.val(args...)
        else
            primal = nothing
        end
        return AugmentedReturn(primal, nothing, nothing)
    end

    function reverse(::Config, ::Const{typeof(wait)}, ::Type{<:Const}, tape, arg)
        closure = cast(fn, arg.dval)
        ev = closure()

        wait(rev)

        return ()
    end
end