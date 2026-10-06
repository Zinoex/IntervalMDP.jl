# Seeded, version-stable problem generators for the benchmark suite.
#
# All randomness comes from `StableRNG(seed)` (StableRNGs.jl), whose stream is
# guaranteed not to change across Julia versions, so a case name always maps to
# the same model. The seed of a case is `case_seed(name)` (CRC32c of the name).
#
# Ambiguity sets are built column by column from a random reference distribution
# `r` over the support (normalised to sum 1):
#   lower = r .* (0.5 + 0.4u₁)        (sum ≤ 0.9 < 1)
#   upper = min.(r .* (1.1 + 0.4u₂), 1) (sum ≥ 1.1 > 1)
# so every column is a valid, non-degenerate interval ambiguity set with a budget
# 1 - sum(lower) ≈ 0.3 that is spread over roughly half of the support.

using StableRNGs, CRC32c, SparseArrays, Random
using IntervalMDP

case_seed(name::AbstractString) = UInt64(crc32c(name))

# Fill `lower`/`gap` for one column given a support length.
@inline function _fill_column!(rng, lower, gap, k)
    s = 0.0
    @inbounds for i in 1:k
        r = rand(rng) + 1e-3
        lower[i] = r
        s += r
    end
    @inbounds for i in 1:k
        r = lower[i] / s
        l = r * (0.5 + 0.4 * rand(rng))
        u = min(r * (1.1 + 0.4 * rand(rng)), 1.0)
        lower[i] = l
        gap[i] = u - l
    end
    return nothing
end

"""
    dense_interval_sets(rng, n, ncols, T)

`n × ncols` dense interval ambiguity sets (full support).
"""
function dense_interval_sets(rng, n, ncols, ::Type{T} = Float64) where {T}
    lower = Matrix{T}(undef, n, ncols)
    gap = Matrix{T}(undef, n, ncols)
    lcol = Vector{Float64}(undef, n)
    gcol = Vector{Float64}(undef, n)
    for j in 1:ncols
        _fill_column!(rng, lcol, gcol, n)
        @views lower[:, j] .= T.(lcol)
        @views gap[:, j] .= T.(gcol)
    end
    return IntervalAmbiguitySets(lower, gap)
end

# Sample `k` distinct sorted indices from 1:n into `out` (k ≪ n: rejection; else shuffle).
function _sample_support!(rng, out, n, k, buf)
    if 4k <= n
        cnt = 0
        while cnt < k
            i = rand(rng, 1:n)
            dup = false
            @inbounds for t in 1:cnt
                if out[t] == i
                    dup = true
                    break
                end
            end
            if !dup
                cnt += 1
                out[cnt] = i
            end
        end
    else
        # partial Fisher–Yates on buf = 1:n
        @inbounds for i in 1:n
            buf[i] = i
        end
        @inbounds for t in 1:k
            j = rand(rng, t:n)
            buf[t], buf[j] = buf[j], buf[t]
            out[t] = buf[t]
        end
    end
    sort!(view(out, 1:k))
    return out
end

"""
    sparse_interval_sets(rng, n, ncols, k, T)

`n × ncols` sparse interval ambiguity sets with exactly `k` non-zeros per column.
`lower` and `gap` share the same `colptr`/`rowval` arrays (Int32 indices).
"""
function sparse_interval_sets(rng, n, ncols, k, ::Type{T} = Float64) where {T}
    k = min(k, n)
    colptr = Int32[1 + k * (j - 1) for j in 1:(ncols + 1)]
    rowval = Vector{Int32}(undef, k * ncols)
    lnz = Vector{T}(undef, k * ncols)
    gnz = Vector{T}(undef, k * ncols)
    idx = Vector{Int}(undef, k)
    buf = Vector{Int}(undef, n)
    lcol = Vector{Float64}(undef, k)
    gcol = Vector{Float64}(undef, k)
    for j in 1:ncols
        _sample_support!(rng, idx, n, k, buf)
        _fill_column!(rng, lcol, gcol, k)
        off = k * (j - 1)
        @inbounds for t in 1:k
            rowval[off + t] = idx[t]
            lnz[off + t] = T(lcol[t])
            gnz[off + t] = T(gcol[t])
        end
    end
    lower = SparseMatrixCSC{T, Int32}(n, ncols, colptr, rowval, lnz)
    gap = SparseMatrixCSC{T, Int32}(n, ncols, colptr, rowval, gnz)
    return IntervalAmbiguitySets(lower, gap)
end

"""
    random_imdp(rng; n, actions, storage, nnz, T)

Interval MDP with `n` states and `actions` actions per state (all available).
`storage = :dense` or `:sparse` (`nnz` non-zeros per column).
"""
function random_imdp(rng; n, actions, storage = :dense, nnz = 0, T = Float64)
    sets = if storage === :dense
        dense_interval_sets(rng, n, n * actions, T)
    else
        sparse_interval_sets(rng, n, n * actions, nnz, T)
    end
    return IntervalMarkovDecisionProcess(sets, actions)
end

"""
    random_fimdp(rng; nvars, nvals, actions, support, T)

Factored IMDP with `nvars` state variables of `nvals` values each. Every marginal
depends on all state variables and the (single) action variable. `support =
nothing` gives dense marginals; an integer `k` gives sparse marginals with `k`
non-zeros per column.
"""
function random_fimdp(rng; nvars, nvals, actions, support = nothing, T = Float64)
    state_vars = ntuple(_ -> nvals, nvars)
    action_vars = (actions,)
    ncols = nvals^nvars * actions
    state_indices = ntuple(identity, nvars)
    marginals = ntuple(nvars) do _
        sets = if isnothing(support)
            dense_interval_sets(rng, nvals, ncols, T)
        else
            sparse_interval_sets(rng, nvals, ncols, support, T)
        end
        Marginal(sets, state_indices, (1,), state_vars, action_vars)
    end
    return FactoredRobustMarkovDecisionProcess(state_vars, action_vars, marginals)
end

"""
    random_states(rng, shape, frac; exclude = Set())

A sorted list of `CartesianIndex`es covering `frac` of the states of `shape`.
"""
function random_states(rng, shape, frac; exclude = Set{CartesianIndex{length(shape)}}())
    all_states = vec(collect(CartesianIndices(shape)))
    cand = [s for s in all_states if s ∉ exclude]
    m = max(1, round(Int, frac * length(all_states)))
    sel = randperm(rng, length(cand))[1:m]
    return sort(cand[sel])
end

"""
    random_dfa_product(rng, mdp)

Product of `mdp` with a fixed 4-state DFA over AP = {a, b} (labels "", a, b, ab):
state 1 (init) --a--> 2 --b--> 3 (accepting, absorbing); 1 --b--> 4; 4 --a--> 1.
States are labelled at random: "" 40%, a 20%, b 20%, ab 20%. Reach set = DFA state 3.
"""
function random_dfa_product(rng, mdp)
    # rows: labels "", "a", "b", "ab"; columns: DFA states 1..4
    T = Int32[
        1 2 3 4
        2 2 3 1
        4 3 3 4
        2 3 3 4
    ]
    dfa = DFA(TransitionFunction(T), 1, ["a", "b"])
    n = num_states(mdp)
    labels = Vector{Int32}(undef, n)
    for i in 1:n
        u = rand(rng)
        labels[i] = u < 0.4 ? 1 : u < 0.6 ? 2 : u < 0.8 ? 3 : 4
    end
    lf = DeterministicLabelling(labels)
    return ProductProcess(mdp, dfa, lf), 3
end
