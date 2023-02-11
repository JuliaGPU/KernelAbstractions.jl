module EnzymeExt
    using EnzymeCore
    using Enzyme
    using EnzymeCore.EnzymeRules
    import KernelAbstractions: Kernel, wait, StaticSize, launch_config, __groupsize, __groupindex, blocks, mkcontext, CPUEvent, CPU, construct

    #TODO: Keywords are the worst
    #
    EnzymeRules.inactive(::typeof(construct), x...) = nothing

    EnzymeRules.inactive(::typeof(Core._compute_sparams), x...) = nothing
    EnzymeRules.inactive(::typeof(StaticSize), x...) = nothing

    function first_type(::Type{NamedTuple{A, B}}) where {A,B}
        first_type(B)
    end
    function first_type(::Type{T}) where {T<:Tuple}
        T.parameters[1]
    end

    function EnzymeRules.augmented_primal(config::Config, func::Const{<:Kernel}, ::Event, args...; ndrange = nothing, workgroupsize = nothing, dependencies = nothing, progress = nothing) where Event
        # Pre-allocate tape according to ndrange... * inner_tape
        kernel = func.val
        f = kernel.f

        ndrange, workgroupsize, iterspace, dynamic = launch_config(kernel, ndrange, workgroupsize)
        block = first(blocks(iterspace))

        ctx = mkcontext(kernel, block, ndrange, iterspace, dynamic)
        ctxTy = typeof(ctx) # Base._return_type(mkcontext, Tuple{typeof(kernel), typeof(block), typeof(ndrange), typeof(iterspace), typeof(dynamic)})
        tt′ = Tuple{Const{ctxTy}, map(Core.typeof, args)...}

        # TODO autodiff_deferred on the func.val
        # TODO modified between use config info
        forward, reverse = Enzyme.Compiler.thunk(f, #=df=#nothing, Const{Nothing}, tt′,  Val(Enzyme.API.DEM_ReverseModePrimal), Val(width(config)), #=ModifiedBetween=#Val(true))
        TapeType = first_type(Base._return_type(forward, tt′))

        subtape = Array{TapeType}(undef, __groupsize(ctx))

        function fwd(ctx, subtape, args...)
            subtape[__groupindex(ctx)] = forward(Const(ctx), args...)[1]
            return nothing
        end

        function rev(ctx, subtape, args...)
            reverse(Const(ctx), args..., subtape[__groupindex(ctx)])
            return nothing
        end

        aug_kernel = similar(kernel, fwd)
        rev_kernel = similar(kernel, rev)

        primal = aug_kernel(subtape, args...; ndrange, workgroupsize, dependencies, progress)::CPUEvent

        tape = Ref{CPUEvent}()
        @assert dependencies === nothing
        function reverse_launch()
            # TODO dependencies ?
            tape[] = rev_kernel(subtape, args...; ndrange, workgroupsize, progress)
            nothing
        end
        if !EnzymeRules.needs_primal(config)
            primal = nothing
        end
        shadow = CPUEvent(Task(reverse_launch))
        return AugmentedReturn(primal, shadow, tape)
    end

    function EnzymeRules.reverse(::Config, ::Const{<:Kernel}, ::Type{<:EnzymeCore.Annotation}, tape, args...)
        wait(tape[])
        return ()
    end

    function EnzymeRules.augmented_primal(config::Config, func::Const{typeof(wait)}, RT, ::Const{CPU}, arg::Duplicated{CPUEvent})
        primal = func.val(arg.val)
        if !EnzymeRules.needs_primal(config)
            primal = nothing
        end
        return AugmentedReturn(primal, nothing, nothing)
    end

    function EnzymeRules.reverse(::Config, ::Const{typeof(wait)}, ::Type{<:Const}, tape, ::Const{CPU}, arg::Duplicated{CPUEvent})
        # This duplicated is invalid, it contains the fake shadow
        closure = arg.dval.task.code
        closure()
        return ()
    end
end
