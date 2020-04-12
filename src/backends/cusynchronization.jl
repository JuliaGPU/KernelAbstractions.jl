module CuSynchronization

using CUDAnative.LLVM
using CUDAnative.LLVM.Interop
using CUDAnative

@generated function unsafe_volatile_load(ptr::Ptr{T}) where T
    eltyp = convert(LLVMType, T)
    T_ptr = convert(LLVMType, Ptr{T})
    T_actual_ptr = LLVM.PointerType(eltyp)

    # create a function
    param_types = [T_ptr]
    llvm_f, _ = create_function(eltyp, param_types)

    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        ptr = inttoptr!(builder, parameters(llvm_f)[1], T_actual_ptr)
        ld = load!(builder, ptr)
        LLVM.API.LLVMSetVolatile(LLVM.ref(ld), LLVM.True)
        ret!(builder, ld)
    end

    call_function(llvm_f, T, Tuple{Ptr{T}}, :(ptr,))
end

@generated function unsafe_volatile_store!(ptr::Ptr{T}, val::T) where T
    eltyp = convert(LLVMType, T)
    T_ptr = convert(LLVMType, Ptr{T})
    T_actual_ptr = LLVM.PointerType(eltyp)

    # create a function
    param_types = [T_ptr, eltyp]
    llvm_f, _ = create_function(LLVM.VoidType(JuliaContext()), param_types)

    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        ptr = inttoptr!(builder, parameters(llvm_f)[1], T_actual_ptr)
        val = parameters(llvm_f)[2]
        st = store!(builder, val, ptr)
        LLVM.API.LLVMSetVolatile(LLVM.ref(st), LLVM.True)
        ret!(builder)
    end

    call_function(llvm_f, Cvoid, Tuple{Ptr{T}, T}, :(ptr, val))
end


import CUDAnative: DevicePtr

struct Semaphore{T, AS}
    sem::DevicePtr{T, AS}
    value::T
end

@inline function Base.getindex(sem::Semaphore{T}) where T
    ptr = Base.unsafe_convert(Ptr{T}, sem.sem)
    unsafe_volatile_load(ptr)
end

@inline function Base.setindex!(sem::Semaphore{T}, val) where T
    ptr = Base.unsafe_convert(Ptr{T}, sem.sem)
    unsafe_volatile_store!(ptr, convert(T, val))
end

@inline function wait(sem::Semaphore{T}) where T
    while true
        if sem[] == sem.value
            break
        end
        threadfence_block()
    end
    sem[] = 2 % T # finalize
    threadfence_system()
    return nothing
end

end # CuSynchronization
