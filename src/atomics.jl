###
# Atomics
###

export atomic_add!, atomic_sub!, atomic_and!, atomic_or!, atomic_xor!,
       atomic_min!, atomic_max!, atomic_inc!, atomic_dec!, atomic_xchg!,
       atomic_op!, atomic_cas!

# helper functions for inc(rement) and dec(rement)
function dec(a::T,b::T) where T
    ((a == 0) | (a > b)) ? b : (a-T(1))
end

function inc(a::T,b::T) where T
    (a >= b) ? T(0) : (a+T(1))
end

# arithmetic, bitwise, min/max, and inc/dec operations
const ops = Dict(
    :atomic_add!   => +,
    :atomic_sub!   => -,
    :atomic_and!   => &,
    :atomic_or!    => |,
    :atomic_xor!   => ⊻,
    :atomic_min!   => min,
    :atomic_max!   => max,
    :atomic_inc!   => inc,
    :atomic_dec!   => dec,
)

# Note: the type T prevents type convertion (for example, Float32 -> 64)
#       can lead to errors if b is chosen to be of a different, compatible type
for (name, op) in ops
    @eval @inline function $name(ptr::Ptr{T}, b::T) where T
        Core.Intrinsics.atomic_pointermodify(ptr::Ptr{T}, $op, b::T, :monotonic)
    end
end

"""
    atomic_cas!(ptr::Ptr{T}, cmp::T, val::T)

This is an atomic Compare And Swap (CAS).
It reads the value `old` located at address `ptr` and compare with `cmp`.
If `old` equals `cmp`, it stores `val` at the same address.
Otherwise, doesn't change the value `old`.
These operations are performed in one atomic transaction.
The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.
Additionally, on GPU hardware with compute capability 7.0+, values of type UInt16 are supported.
Also: atomic operations for the CPU requires a Julia version of 1.7.0 or above.
"""
function atomic_cas!(ptr::Ptr{T}, old::T, new::T) where T
    Core.Intrinsics.atomic_pointerreplace(ptr, old, new, :acquire_release, :monotonic)
end

"""
    atomic_xchg!(ptr::Ptr{T}, val::T)

This is an atomic exchange.
It reads the value `old` located at address `ptr` and stores `val` at the same address.
These operations are performed in one atomic transaction. The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.
Also: atomic operations for the CPU requires a Julia version of 1.7.0 or above.
"""
function atomic_xchg!(ptr::Ptr{T}, b::T) where T
    Core.Intrinsics.atomic_pointerswap(ptr::Ptr{T}, b::T, :monotonic)
end

"""
    atomic_op!(ptr::Ptr{T}, val::T)

This is an arbitrary atomic operation.
It reads the value `old` located at address `ptr` and uses `val` in the operation `op` (defined elsewhere)
These operations are performed in one atomic transaction. The function returns `old`.

This function is somewhat experimental.
Also: atomic operations for the CPU requires a Julia version of 1.7.0 or above.
"""
function atomic_op!(ptr::Ptr{T}, op, b::T) where T
    Core.Intrinsics.atomic_pointermodify(ptr::Ptr{T}, op, b::T, :monotonic)
end

# Other Documentation

"""
    atomic_add!(ptr::Ptr{T}, val::T)

This is an atomic addition.
It reads the value `old` located at address `ptr`, computes `old + val`, and stores the result back to memory at the same address.
These operations are performed in one atomic transaction.
The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32, UInt64, and Float32.
Additionally, on GPU hardware with compute capability 6.0+, values of type Float64 are supported.
Also: atomic operations for the CPU requires a Julia version of 1.7.0 or above.
"""
atomic_add!

"""
    atomic_sub!(ptr::Ptr{T}, val::T)

This is an atomic subtraction.
It reads the value `old` located at address `ptr`, computes `old - val`, and stores the result back to memory at the same address.
These operations are performed in one atomic transaction.
The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.
Also: atomic operations for the CPU requires a Julia version of 1.7.0 or above.
"""
atomic_sub!

"""
    atomic_and!(ptr::Ptr{T}, val::T)

This is an atomic and.
It reads the value `old` located at address `ptr`, computes `old & val`, and stores the result back to memory at the same address.
These operations are performed in one atomic transaction.
The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.
Also: atomic operations for the CPU requires a Julia version of 1.7.0 or above.
"""
atomic_and!

"""
    atomic_or!(ptr::Ptr{T}, val::T)

This is an atomic or.
It reads the value `old` located at address `ptr`, computes `old | val`, and stores the result back to memory at the same address.
These operations are performed in one atomic transaction.
The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.
Also: atomic operations for the CPU requires a Julia version of 1.7.0 or above.
"""
atomic_or!

"""
    atomic_xor!(ptr::Ptr{T}, val::T)

This is an atomic xor.
It reads the value `old` located at address `ptr`, computes `old ⊻ val`, and stores the result back to memory at the same address.
These operations are performed in one atomic transaction.
The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.
Also: atomic operations for the CPU requires a Julia version of 1.7.0 or above.
"""
atomic_xor!

"""
    atomic_min!(ptr::Ptr{T}, val::T)

This is an atomic min.
It reads the value `old` located at address `ptr`, computes `min(old, val)`, and st ores the result back to memory at the same address.
These operations are performed in one atomic transaction.
The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.
Also: atomic operations for the CPU requires a Julia version of 1.7.0 or above.
"""
atomic_min!

"""
    atomic_max!(ptr::Ptr{T}, val::T)

This is an atomic max.
It reads the value `old` located at address `ptr`, computes `max(old, val)`, and st ores the result back to memory at the same address.
These operations are performed in one atomic transaction.
The function returns `old`.

This operation is supported for values of type Int32, Int64, UInt32 and UInt64.
Also: atomic operations for the CPU requires a Julia version of 1.7.0 or above.
"""
atomic_max!

"""
    atomic_inc!(ptr::Ptr{T}, val::T)

This is an atomic increment function that counts up to a certain number before starting again at 0.
It reads the value `old` located at address `ptr`, computes `((old >= val) ? 0 : (o ld+1))`, and stores the result back to memory at the same address.
These three operations are performed in one atomic transaction.
The function returns `old`.

This operation is only supported for values of type Int32.
Also: atomic operations for the CPU requires a Julia version of 1.7.0 or above.
"""
atomic_inc!

"""
    atomic_dec!(ptr::Ptr{T}, val::T)

This is an atomic decrement function that counts down to 0 from a defined value `val`.
It reads the value `old` located at address `ptr`, computes `(((old == 0) | (old > val)) ? val : (old-1))`, and stores the result back to memory at the same address.
These three operations are performed in one atomic transaction.
The function returns `old`.

This operation is only supported for values of type Int32.
Also: atomic operations for the CPU requires a Julia version of 1.7.0 or above.
"""
atomic_dec!
