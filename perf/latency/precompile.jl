# The time to precompile KernelAbstractions, whose dependencies have been precompiled
pkg = Base.identify_package("KernelAbstractions")
t = @elapsed Base.compilecache(pkg)
println(t)
