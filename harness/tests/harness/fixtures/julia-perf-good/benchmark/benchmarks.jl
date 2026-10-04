# Reproducible benchmark entry point (evidence, not a unit test).
# Run baseline and change with the same command and environment, e.g.
#   git checkout <baseline>; julia --project=benchmark benchmark/benchmarks.jl > base.txt
#   git checkout <change>;   julia --project=benchmark benchmark/benchmarks.jl > change.txt
using JuliaPerfGood
r = ones(256); P = fill(1/256, 256, 256); v = ones(256)
bellman_step(r, P, v)
t = minimum(@elapsed(bellman_step(r, P, v)) for _ in 1:100)
println("bellman_step n=256 min_time_s=", t)
