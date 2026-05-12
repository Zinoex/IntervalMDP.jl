###################################
# Custom Iterator Implementations #
###################################

abstract type AbstractIterator end
function Base.length(iter::AbstractIterator) end
function Base.firstindex(iter::AbstractIterator) end
function Base.lastindex(iter::AbstractIterator) end
function Base.getindex(iter::AbstractIterator, i) end
function Base.iterate(iter::AbstractIterator) end
function Base.iterate(iter::AbstractIterator, state) end

###################################
# Sequence shape trait             #
###################################
#
# An update-sequence iterator yields either bare states `s`
# (`StateUpdateSequence`) or state-action pairs `(a, s)`
# (`StateActionUpdateSequence`). `bellman_update!` dispatches on this trait,
# so the inner loop structure is determined by the sampling strategy, not by
# the model/workspace type.

abstract type SequenceShape end
struct StateUpdateSequence <: SequenceShape end
struct StateActionUpdateSequence <: SequenceShape end

# Default: today's iterators all yield (a, s) pairs.
sequence_shape(::AbstractIterator) = StateActionUpdateSequence()
#
# Default implementations work for any iterable that yields either `s` or
# `(a, s)` pairs. Specialized overrides for our concrete iterator types
# avoid repeated set construction.

# `touched_states` returns a `Set{CartesianIndex}` by default. Callers that
# want an ordered walk can `sort!(collect(...))` — the partial-sweep
# relax logic doesn't care about order.
touched_states(seq) = _touched_states(sequence_shape(seq), seq)
_touched_states(::StateUpdateSequence, seq) = Set(seq)
_touched_states(::StateActionUpdateSequence, seq) = Set(s for (_, s) in seq)

# Project an update sequence down to its unique-state set, preserving order.
# Used by V-shape consumers (e.g. gsrdp's `bellman_update!`) that can't
# honour per-(s, a) sampling without a Q-buffer; the resulting state set
# triggers a full action sweep at each visited state.
project_to_state_sequence(seq) = _project_to_state_sequence(sequence_shape(seq), seq)
_project_to_state_sequence(::StateUpdateSequence, seq) = seq
function _project_to_state_sequence(::StateActionUpdateSequence, seq)
    return StateIterator(unique(s for (_, s) in seq))
end

# Partition: default chunks by index range over `eachindex(seq)`. For our
# index-based `AbstractIterator`s this is O(1) per chunk via `SubIterator`.
# Non-indexable fallback collects — callers should prefer indexable
# iterators for large sweeps.
function partition(seq, nchunks::Integer)
    nchunks <= 1 && return Any[seq]
    n = length(seq)
    n == 0 && return Any[seq]
    chunks = Any[]
    chunk_size, rem = divrem(n, nchunks)
    start = firstindex(seq)
    for i in 1:nchunks
        extra = i <= rem ? 1 : 0
        stop = start + chunk_size + extra - 1
        stop < start && continue
        push!(chunks, _view_iter(seq, start:stop))
        start = stop + 1
    end
    return chunks
end

# `_view_iter` returns a lightweight view over `seq[range]`. Concrete
# iterators that admit O(1) slicing should override this; the fallback
# materializes a `Vector` of the iterator's element type.
_view_iter(seq, range) = [seq[i] for i in range]

struct ProductIterator{AI, SI} <: AbstractIterator
    A::AI
    S::SI
    nA::Int
    nS::Int

    function ProductIterator(A, S)
        nA = length(A)
        nS = length(S)
        new{typeof(A), typeof(S)}(A, S, nA, nS)
    end
end

Base.length(iter::ProductIterator) = iter.nA * iter.nS
Base.firstindex(iter::ProductIterator) = (firstindex(iter.S)-1)*iter.nS + firstindex(iter.A)
Base.lastindex(iter::ProductIterator) = (lastindex(iter.S)-1)*iter.nS + lastindex(iter.A)
Base.getindex(iter::ProductIterator, i) = begin
    A = iter.A
    S = iter.S

    nA = iter.nA
    nS = iter.nS

    ia = ((i - 1) % nA) + firstindex(A)
    is = ((i - 1) ÷ nA) + firstindex(S)
    return (A[ia], S[is])
end
Base.iterate(iter::ProductIterator) = begin
    (iter.nA == 0 || iter.nS == 0) && return nothing

    A = iter.A
    S = iter.S

    ia = firstindex(A)
    is = firstindex(S)

    return ((A[ia], S[is]), (ia, is))
end

Base.iterate(iter::ProductIterator, state) = begin
    A = iter.A
    S = iter.S

    ia, is = state

    # iterate s as outer loop due to column major order of value function Q(a, s)

    # 1. advance inner loop (actions)
    ia += 1

    if ia > lastindex(A)
        # 2. reset inner loop and advance outer loop (states)
        ia = firstindex(A)
        is += 1
    end

    # 3. loop exit condition
    if is > lastindex(S)
        return nothing
    end

    return ((A[ia], S[is]), (ia, is))
end

struct ZipIterator{AI, SI} <: AbstractIterator
    A::AI
    S::SI
    n::Int

    function ZipIterator(A, S)
        nA = length(A)
        nS = length(S)

        @assert nA == nS "Action and state spaces must have the same length for ZipIterator"

        new{typeof(A), typeof(S)}(A, S, nA)
    end
end

Base.length(iter::ZipIterator) = iter.n
Base.firstindex(iter::ZipIterator) = 1
Base.lastindex(iter::ZipIterator) = iter.n
Base.getindex(iter::ZipIterator, i) = begin
    A = iter.A
    S = iter.S

    return (A[i], S[i])
end

Base.iterate(iter::ZipIterator) = begin
    iter.n == 0 && return nothing

    A = iter.A
    S = iter.S

    i = 1
    return ((A[firstindex(A) - 1 + i], S[firstindex(S) - 1 + i]), i)
end

Base.iterate(iter::ZipIterator, i) = begin
    A = iter.A
    S = iter.S

    i += 1
    if i > iter.n
        return nothing
    end

    return ((A[firstindex(A) - 1 + i], S[firstindex(S) - 1 + i]), i)
end

struct OnPolicyActionIterator <: AbstractIterator
    S::CartesianIndices
    strategy_cache::AbstractStrategyCache

    function OnPolicyActionIterator(
        S::CartesianIndices,
        strategy_cache::AbstractStrategyCache,
    )
        new(S, strategy_cache)
    end
end
Base.length(iter::OnPolicyActionIterator) = length(iter.S)
Base.firstindex(iter::OnPolicyActionIterator) = firstindex(iter.S)
Base.lastindex(iter::OnPolicyActionIterator) = lastindex(iter.S)
Base.getindex(iter::OnPolicyActionIterator, i) = CartesianIndex(iter.strategy_cache[i])
Base.iterate(iter::OnPolicyActionIterator) = begin
    length(iter) == 0 && return nothing

    S = iter.S
    strategy_cache = iter.strategy_cache

    i = firstindex(S)
    return (CartesianIndex(strategy_cache[i]), i)
end
Base.iterate(iter::OnPolicyActionIterator, i) = begin
    S = iter.S
    strategy_cache = iter.strategy_cache

    i += 1
    if i > lastindex(S)
        return nothing
    end

    return (CartesianIndex(strategy_cache[i]), i)
end

struct GivenSequenceIterator{NA, NS, T} <: AbstractIterator
    sequence::Vector{Tuple{NTuple{NA, T}, NTuple{NS, T}}}
end

Base.length(iter::GivenSequenceIterator) = length(iter.sequence)
Base.firstindex(iter::GivenSequenceIterator) = firstindex(iter.sequence)
Base.lastindex(iter::GivenSequenceIterator) = lastindex(iter.sequence)
Base.getindex(iter::GivenSequenceIterator, i) = begin
    a, s = getindex(iter.sequence, i)

    return (CartesianIndex(a...), CartesianIndex(s...))
end
Base.iterate(iter::GivenSequenceIterator) = begin
    next = iterate(iter.sequence)

    if next === nothing
        return nothing
    end

    value, index = next
    a, s = value

    return ((CartesianIndex(a...), CartesianIndex(s...)), index)
end

Base.iterate(iter::GivenSequenceIterator, state) = begin
    next = iterate(iter.sequence, state)

    if next === nothing
        return nothing
    end

    value, index = next
    a, s = value

    return ((CartesianIndex(a...), CartesianIndex(s...)), index)
end

# Iterator over states only. Yields `s::CartesianIndex`; used by
# `StateUpdateSequence`-shape samplers that sweep all available actions per
# visited state.
struct StateIterator{SI} <: AbstractIterator
    S::SI
    nS::Int

    function StateIterator(S)
        new{typeof(S)}(S, length(S))
    end
end

Base.length(iter::StateIterator) = iter.nS
Base.firstindex(iter::StateIterator) = firstindex(iter.S)
Base.lastindex(iter::StateIterator) = lastindex(iter.S)
Base.getindex(iter::StateIterator, i) = iter.S[i]
Base.iterate(iter::StateIterator) = iterate(iter.S)
Base.iterate(iter::StateIterator, state) = iterate(iter.S, state)

sequence_shape(::StateIterator) = StateUpdateSequence()

###################################
# Sampling Strategies             #
###################################

abstract type SamplingStrategy end

function sample(::SamplingStrategy, model) end

struct AllSampling <: SamplingStrategy end

default_sampling_strategy() = AllSampling()

sample(::AllSampling, model) = exhaustive_cartesian(model)

sample(::AllSampling, model, strategy_cache::AbstractStrategyCache) =
    exhaustive_cartesian(model, strategy_cache)

# `ProductProcess` wraps a Markov process with a DFA. Iteration over the
# (a, s) update sequence is identical to the underlying MDP — the DFA part
# is handled by `_expectation_helper!(::ProductWorkspace, ...)` which
# splits Vres along the DFA-state axis and recursively dispatches to the
# inner Markov process for each DFA state. The two cache-typed methods
# below are split (rather than one `AbstractStrategyCache` method) so that
# they don't tie with the generic
# `exhaustive_cartesian(model, ::OptimizingStrategyCache)` / `(::NonOptimizingStrategyCache)`
# fallbacks below — Julia would consider both equally specific and report
# an ambiguity.
exhaustive_cartesian(proc::ProductProcess) = exhaustive_cartesian(markov_process(proc))
exhaustive_cartesian(proc::ProductProcess, sc::OptimizingStrategyCache) =
    exhaustive_cartesian(markov_process(proc), sc)
exhaustive_cartesian(proc::ProductProcess, sc::NonOptimizingStrategyCache) =
    exhaustive_cartesian(markov_process(proc), sc)

exhaustive_cartesian(model::FactoredRMDP) = exhaustive_cartesian(model, modeltype(model))
exhaustive_cartesian(model::FactoredRMDP, ::IsIMDP) = ProductIterator(
    CartesianIndices(action_shape(model)),
    CartesianIndices(source_shape(model)),
)
# Factored interval MDPs (N marginals). `ProductIterator` works with any
# N-dim `CartesianIndices`, so the iterator is identical in form — the
# difference shows up in how the downstream `_expectation_helper!` interprets
# the multi-dimensional (a, s) indices against each marginal's support.
exhaustive_cartesian(model::FactoredRMDP, ::IsFIMDP) = ProductIterator(
    CartesianIndices(action_shape(model)),
    CartesianIndices(source_shape(model)),
)
exhaustive_cartesian(model::IntervalAmbiguitySets) = ProductIterator(
    CartesianIndices(action_shape(model)),
    CartesianIndices(source_shape(model)),
)

exhaustive_cartesian(model, strategy_cache::OptimizingStrategyCache) =
    exhaustive_cartesian(model)
exhaustive_cartesian(model::FactoredRMDP, strategy_cache::NonOptimizingStrategyCache) =
    exhaustive_cartesian(model, modeltype(model), strategy_cache)

function exhaustive_cartesian(
    model::FactoredRMDP,
    ::IsIMDP,
    strategy_cache::NonOptimizingStrategyCache,
)
    S = CartesianIndices(source_shape(model))
    A = OnPolicyActionIterator(S, strategy_cache)

    return ZipIterator(A, S)
end

function exhaustive_cartesian(
    model::IntervalAmbiguitySets,
    strategy_cache::NonOptimizingStrategyCache,
)
    S = CartesianIndices(source_shape(model))
    A = OnPolicyActionIterator(S, strategy_cache)

    return ZipIterator(A, S)
end

# State-sweep sampler. Yields bare states; the inner loop of `bellman_update!`
# is expected to sweep all available actions per visited state.
struct AllStatesSweep <: SamplingStrategy end

sample(::AllStatesSweep, model) = exhaustive_state_sweep(model)
sample(::AllStatesSweep, model, ::AbstractStrategyCache) = exhaustive_state_sweep(model)

exhaustive_state_sweep(proc::ProductProcess) = exhaustive_state_sweep(markov_process(proc))
exhaustive_state_sweep(model::FactoredRMDP) =
    StateIterator(CartesianIndices(source_shape(model)))
exhaustive_state_sweep(model::IntervalAmbiguitySets) =
    StateIterator(CartesianIndices(source_shape(model)))

# Random-subset sampler: yields `k` independent uniform samples of (a, s)
# pairs per iteration. Primarily useful with
# `GeneralizedSamplingbasedRobustDynamicProgramming` — only visited states
# get their V and strategy relaxed; unvisited states retain V_prev.
struct RandomSubsetStateActions <: SamplingStrategy
    k::Int
end

struct RandomSubsetState <: SamplingStrategy
    k::Int
end

struct RandomSubsetStateActionIterator{NA, NS} <: AbstractIterator
    pairs::Vector{Tuple{CartesianIndex{NA}, CartesianIndex{NS}}}
end

Base.length(iter::RandomSubsetStateActionIterator) = length(iter.pairs)
Base.firstindex(iter::RandomSubsetStateActionIterator) = firstindex(iter.pairs)
Base.lastindex(iter::RandomSubsetStateActionIterator) = lastindex(iter.pairs)
Base.getindex(iter::RandomSubsetStateActionIterator, i) = iter.pairs[i]
Base.iterate(iter::RandomSubsetStateActionIterator) = iterate(iter.pairs)
Base.iterate(iter::RandomSubsetStateActionIterator, state) = iterate(iter.pairs, state)

sample(ss::RandomSubsetStateActions, model) = random_subset_state_action_sample(ss.k, model)
sample(ss::RandomSubsetStateActions, model, ::AbstractStrategyCache) =
    random_subset_state_action_sample(ss.k, model)

function random_subset_state_action_sample(k::Int, model)
    A = CartesianIndices(action_shape(model))
    S = CartesianIndices(source_shape(model))
    pairs = Vector{Tuple{eltype(A), eltype(S)}}(undef, k)
    @inbounds for i in 1:k
        pairs[i] = (rand(A), rand(S))
    end
    return RandomSubsetStateActionIterator(pairs)
end

struct RandomSubsetStateIterator{NS} <: AbstractIterator
    states::Vector{CartesianIndex{NS}}
end

Base.length(iter::RandomSubsetStateIterator) = length(iter.states)
Base.firstindex(iter::RandomSubsetStateIterator) = firstindex(iter.states)
Base.lastindex(iter::RandomSubsetStateIterator) = lastindex(iter.states)
Base.getindex(iter::RandomSubsetStateIterator, i) = iter.states[i]
Base.iterate(iter::RandomSubsetStateIterator) = iterate(iter.states)
Base.iterate(iter::RandomSubsetStateIterator, state) = iterate(iter.states, state)

sequence_shape(::RandomSubsetStateIterator) = StateUpdateSequence()

sample(ss::RandomSubsetState, model) = random_subset_state_sample(ss.k, model)
sample(ss::RandomSubsetState, model, ::AbstractStrategyCache) =
    random_subset_state_sample(ss.k, model)

random_subset_state_sample(k::Int, proc::ProductProcess) =
    random_subset_state_sample(k, markov_process(proc))

function random_subset_state_sample(k::Int, model)
    S = CartesianIndices(source_shape(model))
    states = Vector{eltype(S)}(undef, k)
    @inbounds for i in 1:k
        states[i] = rand(S)
    end
    return RandomSubsetStateIterator(states)
end

struct GivenSequence <: SamplingStrategy end

function sample(
    ::GivenSequence,
    model,
    sequence::Vector{Tuple{NTuple{N, T}, NTuple{M, T}}}, # each element: (state_tuple, action_tuple
) where {N, M, T <: Integer}
    return custom_sequence(model, sequence)
end

function custom_sequence(
    model::FactoredRMDP,
    sequence::Vector{Tuple{NTuple{N, T}, NTuple{M, T}}},
)::AbstractVector{Tuple{CartesianIndex{N}, CartesianIndex{M}}} where {N, M, T <: Integer}

    # Precompute model shapes
    shape_s = source_shape(model)   # state shape tuple
    shape_a = action_shape(model)   # action shape tuple

    # Validate each state-action pair
    for (s, a) in sequence
        # check all entries >= 1
        @assert all(x -> x >= 1, s)
        @assert all(x -> x >= 1, a)

        # check each entry within bounds
        @assert all((xi, yi) -> xi <= yi, zip(s, shape_s))
        @assert all((xi, yi) -> xi <= yi, zip(a, shape_a))
    end

    return GivenSequenceIterator(sequence)
end

# TODO: 1. random sampling of states, with or without replacement, with or without weighting (e.g. based on current value function)
# TODO:     - subset of states each iteration?
# TODO:     - one state per iteration?
# TODO:
# TODO: 2. (epsilon) greedy on policy trajectory simulation
# TODO: 3. BRTDP gap based trajectory simulation
# TODO: 

### Robust Value Iteration
# RobustVI does a full state-outer sweep — yields bare states so
# `expectation_v!` dispatches to the state-outer path that uses
# `workspace.actions` + `extract_strategy!` (no Q-array materialization).
sampling_strategy(alg::RobustValueIteration) = AllStatesSweep()

### Generalized Sampling-based Robust Dynamic Programming
sampling_strategy(alg::GeneralizedSamplingbasedRobustDynamicProgramming) =
    alg.sampling_strategy
