# Run in the back ends' test environments, as `Pkg.test`'s `test_fn`: check that they use this
# checkout of KernelAbstractions and KernelInterface, and not one that a back end's `[sources]`
# point to, which Julia 1.12+ honours.
let checkout = dirname(@__DIR__)
    for (name, uuid, dir) in (
            ("KernelAbstractions", "63c18a36-062a-441e-b654-da1e3ab1ce7c", checkout),
            ("KernelInterface", "4ee993da-d684-4d17-a7dd-4e58e78d92bf", joinpath(checkout, "lib", "KernelInterface")),
        )
        path = Base.locate_package(Base.PkgId(Base.UUID(uuid), name))
        expected = joinpath(dir, "src", "$name.jl")
        path !== nothing && realpath(path) == realpath(expected) ||
            error("The tests would use $name from $path instead of this checkout")
        println("Testing $name from $(dirname(dirname(path)))")
    end
end
