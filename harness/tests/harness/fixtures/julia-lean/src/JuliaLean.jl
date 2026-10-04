module JuliaLean
export bellman_step
"""
    bellman_step(r, P, v)

Fixture-only, dependency-free "Bellman-like" update: r .+ P * v.
Exists purely so harness tests can run `Pkg.test()` offline.
"""
bellman_step(r::AbstractVector, P::AbstractMatrix, v::AbstractVector) = r .+ P * v

end
