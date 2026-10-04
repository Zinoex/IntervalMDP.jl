# Lint-only fixture file (not included by runtests.jl; BenchmarkTools is not a
# dependency). BenchmarkTools trial objects are timing measurements too, so a
# threshold on them in a unit test must be flagged by timing_lint.py.
using BenchmarkTools, Test, JuliaPerfBad
P = ones(2, 2)
b = @benchmark bellman_step(ones(2), P, ones(2))
@test median(b).time < 1e6
m = minimum(run(@benchmarkable bellman_step(ones(2), P, ones(2))))
@test m.time < 1e6
@test (@ballocated bellman_step(ones(2), P, ones(2))) < 10_000   # allocation check: NOT flagged
