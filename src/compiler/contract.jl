for (f, T) in Base.Iterators.product((:add, :mul, :sub), (Float32, Float64))
    name = Symbol("$(f)_float_contract")
    if T === Float32
        llvmt = "float"
    elseif T === Float64
        llvmt = "double"
    end

    #XXX Use LLVM.jl
    ir = """
        %x = f$f contract $llvmt %0, %1
        ret $llvmt %x
    """
    @eval begin
        # the @pure is necessary so that we can constant propagate.
        Base.@pure function $name(a::$T, b::$T)
            @Base._inline_meta
            Base.llvmcall($ir, $T, Tuple{$T, $T}, a, b)
        end
    end
end