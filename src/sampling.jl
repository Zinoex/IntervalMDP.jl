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

"""
    SamplingStrategy

Abstract supertype for sampling strategies. A sampling strategy decides which
states (or `(action, state)` pairs) are relaxed in each iteration of a
sampling-based algorithm such as
[`GeneralizedSamplingbasedRobustDynamicProgramming`](@ref).

Concrete strategies implement `sample(strategy, model)` (and, optionally,
`sample(strategy, model, strategy_cache)`), returning an iterator over the
update sequence for the current iteration. Every concrete strategy falls into
one of seven categories, each a direct subtype of `SamplingStrategy`:
[`AllSamplingStrategy`](@ref), [`RandomSamplingStrategy`](@ref),
[`RoundRobinSamplingStrategy`](@ref), [`TrajectorySamplingStrategy`](@ref),
[`PriorityQueueSamplingStrategy`](@ref), [`GivenSequence`](@ref) (a single
concrete type, no category needed), and [`CompositeSamplingStrategy`](@ref)
(strategies that wrap other strategies).
"""
abstract type SamplingStrategy end

function sample(::SamplingStrategy, model) end

###################################
# Context-requirement trait        #
###################################
# `_gsrdp_sample` (`gsrdp.jl`) prefers the richest `sample` signature a
# strategy supports: `sample(strategy, model, strategy_cache, value_function,
# spec)` over the plain `sample(strategy, model[, strategy_cache])`. Because
# the category hierarchy above is flat (every category is a direct subtype of
# `SamplingStrategy`, not nested under a "needs value function" supertype),
# that preference can't be read off the type hierarchy the way
# `sequence_shape` reads off the *iterator* hierarchy earlier in this file —
# so it's a trait instead, following the same pattern.

abstract type SamplingContextRequirement end

"`sample(strategy, model)` / `sample(strategy, model, strategy_cache)`."
struct NeedsModelOnly <: SamplingContextRequirement end

"`sample(strategy, model, strategy_cache, value_function, spec)`."
struct NeedsValueFunctionAndSpec <: SamplingContextRequirement end

sampling_context_requirement(::SamplingStrategy) = NeedsModelOnly()

###################################
# Lifecycle: reset hook            #
###################################
# Strategies that carry mutable state across iterations (e.g. a round-robin
# cursor) must reset it at the start of every `solve` — `_gsrdp!` calls
# `reset_sampling_strategy!` once, before the first `sample` call, so the
# same algorithm/strategy object can be reused safely across repeated
# `solve` calls (e.g. a warm-up solve followed by many timed samples in a
# benchmark harness, all reusing the same `alg`). Stateless strategies use
# the no-op default below; composites propagate the reset to their children
# via `sub_strategies`.

sub_strategies(::SamplingStrategy) = ()

function reset_sampling_strategy!(ss::SamplingStrategy)
    foreach(reset_sampling_strategy!, sub_strategies(ss))
    return nothing
end

###################################
# 1. All-sampling                  #
###################################

"""
    AllSamplingStrategy <: SamplingStrategy

Abstract supertype for exhaustive sampling strategies that relax every state
(or `(action, state)` pair) each iteration: [`AllSampling`](@ref) and
[`AllStatesSweep`](@ref).
"""
abstract type AllSamplingStrategy <: SamplingStrategy end

"""
    AllSampling()

Exhaustive sampler: relaxes every `(action, state)` pair in the model on each
iteration (a full Cartesian sweep). This is the standard behaviour for full
robust value iteration.
"""
struct AllSampling <: AllSamplingStrategy end

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

"""
    AllStatesSweep()

State-sweep sampler. Yields bare states (rather than `(action, state)` pairs);
the inner loop of `bellman_update!` then sweeps all available actions per
visited state. This is the default sampling strategy for
[`GeneralizedSamplingbasedRobustDynamicProgramming`](@ref).
"""
struct AllStatesSweep <: AllSamplingStrategy end

sample(::AllStatesSweep, model) = exhaustive_state_sweep(model)
sample(::AllStatesSweep, model, ::AbstractStrategyCache) = exhaustive_state_sweep(model)

exhaustive_state_sweep(proc::ProductProcess) = exhaustive_state_sweep(markov_process(proc))
exhaustive_state_sweep(model::FactoredRMDP) =
    StateIterator(CartesianIndices(source_shape(model)))
exhaustive_state_sweep(model::IntervalAmbiguitySets) =
    StateIterator(CartesianIndices(source_shape(model)))

###################################
# 2. Random sampling               #
###################################

"""
    RandomSamplingStrategy <: SamplingStrategy

Abstract supertype for uniform-random sampling strategies:
[`RandomSubsetState`](@ref) and [`RandomSubsetStateActions`](@ref).
"""
abstract type RandomSamplingStrategy <: SamplingStrategy end

"""
    RandomSubsetStateActions(k)

Random-subset sampler: yields `k` independent uniform samples of
`(action, state)` pairs per iteration. Primarily useful with
[`GeneralizedSamplingbasedRobustDynamicProgramming`](@ref) — only visited
states get their value and strategy relaxed; unvisited states retain their
previous value.
"""
struct RandomSubsetStateActions <: RandomSamplingStrategy
    k::Int
end

"""
    RandomSubsetState(k)

Random-subset sampler: yields `k` independent uniform samples of states per
iteration (bare states, so all available actions are swept per visited state).
Like [`RandomSubsetStateActions`](@ref), only visited states are relaxed each
iteration; unvisited states retain their previous value.
"""
struct RandomSubsetState <: RandomSamplingStrategy
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

###################################
# 3. Round-robin sampling          #
###################################

"""
    RoundRobinSamplingStrategy <: SamplingStrategy

Abstract supertype for round-robin sampling strategies: each iteration
advances a persistent cursor through a fixed enumeration of states (or
`(action, state)` pairs) by `k` entries, wrapping around once the cursor
reaches the end. Every state (or pair) is visited on a regular cycle, rather
than probabilistically as in [`RandomSamplingStrategy`](@ref).

Concrete round-robin strategies carry mutable cursor state that persists
across `sample` calls on the same instance — safe because
[`GeneralizedSamplingbasedRobustDynamicProgramming`](@ref) reuses the same
strategy object across the iterations of a single `solve`.
`reset_sampling_strategy!` rewinds the cursor to the start; it is called
automatically once at the start of every `solve`.
"""
abstract type RoundRobinSamplingStrategy <: SamplingStrategy end

"""
    RoundRobinState(k)

Round-robin sampler: cycles through states in `CartesianIndices` order (the
same order [`AllStatesSweep`](@ref) enumerates), yielding the next `k` states
each iteration and wrapping around once every state has been visited. Yields
bare states, so the inner loop of `bellman_update!` sweeps all available
actions per visited state.
"""
struct RoundRobinState <: RoundRobinSamplingStrategy
    k::Int
    cursor::Base.RefValue{Int}

    RoundRobinState(k::Int) = new(k, Ref(0))
end

"""
    RoundRobinStateActions(k)

Round-robin sampler: cycles through `(action, state)` pairs in the same
linear order [`AllSampling`](@ref) enumerates, yielding the next `k` pairs
each iteration and wrapping around once every pair has been visited.
"""
struct RoundRobinStateActions <: RoundRobinSamplingStrategy
    k::Int
    cursor::Base.RefValue{Int}

    RoundRobinStateActions(k::Int) = new(k, Ref(0))
end

reset_sampling_strategy!(ss::RoundRobinSamplingStrategy) = (ss.cursor[] = 0; nothing)

# Next `k` linear indices into `1:n`, wrapping around; advances `cursor` by
# `k` (mod `n`). `n == 0` yields an empty window rather than dividing by zero.
function _round_robin_indices(k::Int, cursor::Base.RefValue{Int}, n::Int)
    n == 0 && return Int[]
    start = cursor[]
    idxs = Vector{Int}(undef, k)
    @inbounds for i in 1:k
        idxs[i] = mod(start + i - 1, n) + 1
    end
    cursor[] = mod(start + k, n)
    return idxs
end

struct RoundRobinStateIterator{NS} <: AbstractIterator
    states::Vector{CartesianIndex{NS}}
end

Base.length(iter::RoundRobinStateIterator) = length(iter.states)
Base.firstindex(iter::RoundRobinStateIterator) = firstindex(iter.states)
Base.lastindex(iter::RoundRobinStateIterator) = lastindex(iter.states)
Base.getindex(iter::RoundRobinStateIterator, i) = iter.states[i]
Base.iterate(iter::RoundRobinStateIterator) = iterate(iter.states)
Base.iterate(iter::RoundRobinStateIterator, state) = iterate(iter.states, state)

sequence_shape(::RoundRobinStateIterator) = StateUpdateSequence()

sample(ss::RoundRobinState, model) = round_robin_state_sample(ss, model)
sample(ss::RoundRobinState, model, ::AbstractStrategyCache) = round_robin_state_sample(ss, model)

round_robin_state_sample(ss::RoundRobinState, proc::ProductProcess) =
    round_robin_state_sample(ss, markov_process(proc))

function round_robin_state_sample(ss::RoundRobinState, model)
    S = CartesianIndices(source_shape(model))
    idxs = _round_robin_indices(ss.k, ss.cursor, length(S))
    return RoundRobinStateIterator([S[i] for i in idxs])
end

struct RoundRobinStateActionIterator{NA, NS} <: AbstractIterator
    pairs::Vector{Tuple{CartesianIndex{NA}, CartesianIndex{NS}}}
end

Base.length(iter::RoundRobinStateActionIterator) = length(iter.pairs)
Base.firstindex(iter::RoundRobinStateActionIterator) = firstindex(iter.pairs)
Base.lastindex(iter::RoundRobinStateActionIterator) = lastindex(iter.pairs)
Base.getindex(iter::RoundRobinStateActionIterator, i) = iter.pairs[i]
Base.iterate(iter::RoundRobinStateActionIterator) = iterate(iter.pairs)
Base.iterate(iter::RoundRobinStateActionIterator, state) = iterate(iter.pairs, state)

sample(ss::RoundRobinStateActions, model) = round_robin_state_action_sample(ss, model)
sample(ss::RoundRobinStateActions, model, ::AbstractStrategyCache) =
    round_robin_state_action_sample(ss, model)

function round_robin_state_action_sample(ss::RoundRobinStateActions, model)
    A = CartesianIndices(action_shape(model))
    S = CartesianIndices(source_shape(model))
    nA, nS = length(A), length(S)
    idxs = _round_robin_indices(ss.k, ss.cursor, nA * nS)
    pairs = Vector{Tuple{eltype(A), eltype(S)}}(undef, length(idxs))
    @inbounds for (j, lin) in enumerate(idxs)
        ia = ((lin - 1) % nA) + firstindex(A)
        is = ((lin - 1) ÷ nA) + firstindex(S)
        pairs[j] = (A[ia], S[is])
    end
    return RoundRobinStateActionIterator(pairs)
end

###################################
# 4. Trajectory sampling           #
###################################

"""
    TrajectorySamplingStrategy <: SamplingStrategy

Abstract supertype for trajectory-based sampling strategies: instead of
sampling states independently, these simulate one or more trajectories
through the model — starting from an initial state, repeatedly picking an
action, realizing a concrete transition distribution via O-maximization
against the upper value function, and sampling a next state from it — and
relax the states visited along the way. A trajectory stops when it reaches a
reach or avoid state, or when the strategy's own [`terminate_sampling`](@ref)
says so (subject to a hard step cap regardless — see [`_trajectory_rollout`]).

A rollout also ends if it leaves the source sub-box (an implicit sink state,
see [`_is_source_state`](@ref)) — the sink is absorbing, and is excluded from
the returned trajectory since it cannot be relaxed.

Concrete subtypes implement four functions — [`num_trajectories`](@ref),
[`terminate_sampling`](@ref), [`action_selection`](@ref),
[`target_state_sampling`](@ref) — and get the rollout, the O-max realization,
initial-state selection, and reach/avoid checking for free (candidates for
those four: epsilon-greedy simulation of the current policy trajectory;
BRTDP-style gap-based trajectory simulation — no concrete strategy is
implemented yet, only this shared skeleton).

Only flat (non-factored) models are supported — the O-max realization
(`_omax_marginal`) requires a single `Marginal`.
"""
abstract type TrajectorySamplingStrategy <: SamplingStrategy end

sampling_context_requirement(::TrajectorySamplingStrategy) = NeedsValueFunctionAndSpec()

sample(ss::TrajectorySamplingStrategy, model, strategy_cache, value_function) =
    sample(ss, model, strategy_cache, value_function, nothing)

###################################
# 4a. Concrete O-max realization   #
###################################
# `state_action_bellman`/`gap_value` (bellman/kernels.jl) already implement
# O-maximization's sort-and-greedily-fill algorithm, but only accumulate the
# scalar expectation `dot(V, p)` — the per-target realized probability `p[i]`
# is computed inline and discarded. Reusing those means depending on
# workspace-internal precomputed state (`permutation`, `budget`) sized for a
# full per-iteration sweep. A trajectory step runs O(1) times per GSRDP
# iteration (not the hot sweep loop), so it's simpler and safer to
# self-contain the same algorithm here instead, off the public
# `IntervalAmbiguitySet` accessors (`lower`, `gap`, `support`).

function _omax_marginal(model)
    ms = marginals(model)
    length(ms) == 1 || throw(
        ArgumentError(
            "trajectory sampling only supports flat (non-factored) models; got $(length(ms)) marginals",
        ),
    )
    return ms[1]
end

_omax_value_order(ambiguity_set, V, upper_bound::Bool) =
    sort(collect(support(ambiguity_set)); by = i -> @inbounds(V[i]), rev = upper_bound)

# Shared greedy fill: walk `order`, adding min(budget, gap[i]) to target i
# until budget is exhausted. `f(i, Δ)` is called for each nonzero fill.
function _omax_fill(f, ambiguity_set, order, budget)
    for i in order
        Δ = min(budget, gap(ambiguity_set, i))
        Δ > zero(Δ) && f(i, Δ)
        budget -= Δ
        budget <= zero(budget) && break
    end
end

"""
    _omax_distribution(ambiguity_set, V, upper_bound) -> Vector

The concrete transition distribution O-maximization realizes for one
`(state, action)`'s `ambiguity_set` against value vector `V` — every target
starts at its lower bound, then targets are filled greedily in `V`-sorted
order (descending / optimistic if `upper_bound`, ascending otherwise) up to
their upper bound until the probability budget `1 - sum(lower)` is
exhausted. Always a full dense `Vector` over every target state, even for a
sparse ambiguity set.
"""
function _omax_distribution(ambiguity_set, V, upper_bound::Bool)
    p = Vector(lower(ambiguity_set))
    order = _omax_value_order(ambiguity_set, V, upper_bound)
    budget = one(eltype(p)) - sum(p)
    _omax_fill(ambiguity_set, order, budget) do i, Δ
        p[i] += Δ
    end
    return p
end

"""
    _omax_expectation(ambiguity_set, V, upper_bound) -> Real

The scalar expectation `dot(V, p)` for the same realized distribution `p`
[`_omax_distribution`](@ref) would build, without materializing the full
vector — for callers (action-value comparisons) that only need the number.
"""
function _omax_expectation(ambiguity_set, V, upper_bound::Bool)
    order = _omax_value_order(ambiguity_set, V, upper_bound)
    budget = one(eltype(V)) - sum(lower(ambiguity_set))
    res = dot(V, lower(ambiguity_set))
    _omax_fill(ambiguity_set, order, budget) do i, Δ
        res += Δ * V[i]
    end
    return res
end

"""
    _omax_best_action(model, jₛ, V, upper_bound; exclude=nothing) -> (action_or_nothing, value_or_nothing)

The available action at state `jₛ` maximizing [`_omax_expectation`](@ref)
against `V`/`upper_bound` — i.e. `argmax_a Q(jₛ, a)` under the O-max
Q-value for that direction. With `exclude`, maximizes among every available
action *except* `exclude`. Returns `(nothing, nothing)` if no candidate
action exists (no actions available at `jₛ`, or `exclude` was the only one).
"""
function _omax_best_action(model, jₛ, V, upper_bound::Bool; exclude = nothing)
    marginal = _omax_marginal(model)
    best_a, best_v = nothing, nothing
    for jₐ in available(model, jₛ)
        (exclude !== nothing && jₐ == exclude) && continue
        v = _omax_expectation(marginal[jₐ, jₛ], V, upper_bound)
        if best_v === nothing || v > best_v
            best_a, best_v = jₐ, v
        end
    end
    return best_a, best_v
end

"""
    _action_uncertainty(model, sp, value_function) -> Real

`V^a(s') = U^{-a_L(s')}(s') - L(s')`, where `a_L(s')` is the action
maximizing the lower-bound Q-value at `s'` and `U^{-a_L}(s')` is the best
upper-bound Q-value at `s'` among the *other* actions — a measure of how
uncertain it still is whether the (lower-bound-)optimal action at `s'`
really is optimal. Returns 0 when `s'` has only one available action (no
alternative to be uncertain about). Shared by
[`TrajectorySampling.ActionUncertaintyTrajectorySampling`](@ref) and
[`PriorityQueueSampling.ActionUncertaintyPriorityQueueSampling`](@ref).
"""
function _action_uncertainty(model, sp, value_function)
    L, U = value_function.lower.current, value_function.upper.current
    # An implicit sink (a target state outside the source sub-box) has no
    # ambiguity-set column and no actions at all, so there is no action to be
    # uncertain about — same reasoning as the single-action case below.
    _is_source_state(model, sp) || return zero(eltype(U))
    a_L, L_sp = _omax_best_action(model, sp, L, false)
    _, U_excl = _omax_best_action(model, sp, U, true; exclude = a_L)
    U_excl === nothing && return zero(eltype(U))   # only one action at sp: no alternative to be uncertain about
    return U_excl - L_sp
end

"""
    _state_indices(model) -> CartesianIndices

`CartesianIndices(source_shape(model))` — every *source* state index, in the
same order [`AllStatesSweep`](@ref) enumerates them.
"""
_state_indices(model) = CartesianIndices(source_shape(model))

"""
    _target_indices(model) -> CartesianIndices

`CartesianIndices(state_values(model))` — every *target* state index. A
superset of [`_state_indices`](@ref): a model may declare `source_dims`
smaller than `state_vars` to leave trailing states implicit ("implicit sink
states", see `FactoredRobustMarkovDecisionProcess`), which are absorbing —
they have no ambiguity-set column and no available actions. Value functions
and the `_omax_distribution` vector are indexed over *this* range.
"""
_target_indices(model) = CartesianIndices(state_values(model))

"""
    _target_state(model, i) -> CartesianIndex

The target state at linear index `i` — how a linear index from
[`_categorical_sample`](@ref) (or from an ambiguity set's `support`) maps
back to a state index. Only equals `CartesianIndex(i)` for single-state-variable
models.
"""
Base.@propagate_inbounds _target_state(model, i) = _target_indices(model)[i]

"""
    _is_source_state(model, s) -> Bool

Whether target state `s` is also a source state — i.e. whether it has
transitions and actions of its own, and an entry in the strategy cache.
False exactly for the implicit sink states described in
[`_target_indices`](@ref); those must never appear in an update sequence,
since `bellman_v!` and the (`source_shape`-sized) strategy cache cannot
index them.
"""
_is_source_state(model, s::CartesianIndex) = s in _state_indices(model)

"""
    _successor_states(model, s) -> Set

`SDS(s) = {s' | ∃a. p(s'|s,a)}` — the union, over every action available at
`s`, of that `(s, a)` ambiguity set's support: every state `s` could
possibly transition to under some action. Used by incremental
priority-queue sampling to find which states' priorities may have gone
stale after `s` was relaxed.

Implicit sink successors (see [`_is_source_state`](@ref)) are excluded: they
are absorbing, so they carry no priority entry that could go stale.
"""
function _successor_states(model, s)
    marginal = _omax_marginal(model)
    T = _target_indices(model)
    succ = Set{eltype(T)}()
    for a in available(model, s)
        for i in support(marginal[a, s])
            sp = T[i]
            _is_source_state(model, sp) && push!(succ, sp)
        end
    end
    return succ
end

###################################
# 4b. Reach/avoid, initial state   #
###################################
# `reach`/`avoid` (specification.jl) are only defined for
# AbstractReachability/AbstractReachAvoid/AbstractSafety — not every
# `Property`. These fall back to "no reach/avoid states" for anything else,
# so trajectory sampling still runs (just never stops via goal/obstacle).

_trajectory_reach_states(prop) = CartesianIndex[]
_trajectory_reach_states(prop::AbstractReachability) = reach(prop)   # AbstractReachAvoid IS-A AbstractReachability

_trajectory_avoid_states(prop) = CartesianIndex[]
_trajectory_avoid_states(prop::AbstractReachAvoid) = avoid(prop)
_trajectory_avoid_states(prop::AbstractSafety) = avoid(prop)

# `initial_states(mp)` elements aren't normalized to `CartesianIndex` at
# model-construction time (could be `Int`, `Tuple`, or `CartesianIndex`).
_to_state_index(s::CartesianIndex) = s
_to_state_index(s::Integer) = CartesianIndex(s)
_to_state_index(s::Tuple) = CartesianIndex(s)

# `AllStates()` (no restriction declared) samples uniformly over every
# state; a concrete initial-states vector samples uniformly among them.
function _trajectory_initial_state(model)
    init = initial_states(model)
    return init isa AllStates ? rand(CartesianIndices(source_shape(model))) :
           _to_state_index(rand(init))
end

###################################
# 4c. The four-function contract   #
###################################

"""
    num_trajectories(strategy::TrajectorySamplingStrategy) -> Int

Number of independent trajectories `sample` rolls out and concatenates per
call. Must be implemented by every concrete subtype.
"""
function num_trajectories end

"""
    terminate_sampling(strategy, current_state, trajectory, value_function, model, spec) -> Bool

Strategy-specific extra stopping condition (e.g. a max length, an
uncertainty/gap threshold) — `trajectory` is the sequence of states visited
so far (including `current_state`, its last entry). The shared rollout ALSO
always stops when `current_state` is a reach or avoid state (checked
independently, before this is called) and enforces its own hard step cap
regardless of what this returns. Must be implemented by every concrete
subtype.
"""
function terminate_sampling end

"""
    action_selection(strategy, current_state, value_function, model, spec) -> action::CartesianIndex

Choose the action to take from `current_state`. `_omax_marginal(model)[a,
current_state]` gives the `(current_state, a)` ambiguity set, and
`_omax_value_order` the O-max value ordering, for strategies that want an
O-max Q-value per available action (`available(model, current_state)`).
Must be implemented by every concrete subtype.
"""
function action_selection end

"""
    target_state_sampling(strategy, current_state, action, probabilities, value_function, model, spec) -> next_state::CartesianIndex

Choose the next state given the concrete O-max transition distribution
`probabilities` (a dense `Vector` over every state, from [`_omax_distribution`](@ref))
computed for `(current_state, action)` against the upper value function.
[`_categorical_sample`](@ref) samples directly from a probability vector,
for strategies that just want that; it yields a linear index, which
[`_target_state`](@ref) maps to the state index this must return. Must be
implemented by every concrete subtype.
"""
function target_state_sampling end

"""
    reverse_trajectory(strategy::TrajectorySamplingStrategy) -> Bool

Whether `sample` reverses each rollout before returning it — goal-first
ordering (the default, `true`) means a Gauss-Seidel-style sweep propagates
the newly-touched goal/obstacle-adjacent values backward through the rest of
the trajectory in the same iteration. Override per strategy for
forward/chronological order instead.
"""
reverse_trajectory(::TrajectorySamplingStrategy) = true

# Hard safety cap, independent of `terminate_sampling` — guarantees a single
# `sample` call can't hang GSRDP if a strategy's own termination logic never
# fires (e.g. an absorbing-free transient region).
_trajectory_max_steps(model) = 10 * num_states(model)

function _trajectory_rollout(ss::TrajectorySamplingStrategy, model, value_function, spec)
    prop = system_property(spec)
    reach_set = Set(_trajectory_reach_states(prop))
    avoid_set = Set(_trajectory_avoid_states(prop))
    V_upper = value_function.upper.current
    max_steps = _trajectory_max_steps(model)

    current = _trajectory_initial_state(model)
    trajectory = [current]
    steps = 0
    while !(current in reach_set) &&
          !(current in avoid_set) &&
          !terminate_sampling(ss, current, trajectory, value_function, model, spec) &&
          steps < max_steps
        a = action_selection(ss, current, value_function, model, spec)
        ambiguity_set = _omax_marginal(model)[a, current]
        probs = _omax_distribution(ambiguity_set, V_upper, true)
        current = target_state_sampling(ss, current, a, probs, value_function, model, spec)
        # An implicit sink is absorbing and has no strategy-cache entry, so
        # the trajectory both ends here and excludes it — `bellman_v!` and the
        # strategy cache can only index source states.
        _is_source_state(model, current) || break
        push!(trajectory, current)
        steps += 1
    end

    return reverse_trajectory(ss) ? reverse(trajectory) : trajectory
end

function sample(ss::TrajectorySamplingStrategy, model, strategy_cache, value_function, spec)
    states = reduce(
        vcat,
        (
            _trajectory_rollout(ss, model, value_function, spec) for
            _ in 1:num_trajectories(ss)
        ),
    )
    return StateIterator(states)
end

"""
    _categorical_sample(probs::AbstractVector) -> Int

Sample a *linear* target index from probability vector `probs` (not required
to be normalized — sampled against `sum(probs)`), via cumulative-sum search.
Available for [`target_state_sampling`](@ref) implementations that just want
to sample directly from the O-max distribution; since those must return a
state index, pass the result through [`_target_state`](@ref).
"""
function _categorical_sample(probs::AbstractVector)
    u = rand() * sum(probs)
    acc = zero(eltype(probs))
    for i in eachindex(probs)
        acc += probs[i]
        acc >= u && return i
    end
    return lastindex(probs)   # floating-point fallback
end

###################################
# 4e. Greedy trajectory sampling   #
###################################
#
# Concrete TrajectorySamplingStrategy algorithms live in this submodule
# (`IntervalMDP.TrajectorySampling.ReachProbabilityTrajectorySampling()` etc.) rather
# than as flat top-level `IntervalMDP.*` names — as more trajectory-sampling
# algorithms are added (BRTDP-gap, epsilon-greedy, ...) this keeps the
# top-level namespace from filling up with many similarly-named,
# trajectory-specific types. The general interface
# (`TrajectorySamplingStrategy`, the four-function contract, the rollout,
# the O-max primitives above) stays in the parent `IntervalMDP` module,
# since `_gsrdp_sample`/`sampling_context_requirement` dispatch on it.

# IntervalMDP.TrajectorySampling: concrete TrajectorySamplingStrategy
# algorithms. All of them share TrajectorySampling.TrajectorySampling's
# behaviour (below) — one trajectory per call, greedy argmax-upper-bound
# action selection, `terminate_sampling` a stub (`false`, for now) — and
# differ only in how they pick the next state given the concrete O-max
# transition distribution:
#
#   * TransitionProbabilityTrajectorySampling — sample directly from it.
#   * ExpectedGapTrajectorySampling — weight by the successor's value gap.
#   * ActionUncertaintyTrajectorySampling — weight by the successor's
#     action-selection uncertainty.
#   * ReachProbabilityTrajectorySampling — weight by the successor's upper
#     (optimistic reach-probability) value.
#
# NOTE: this module and its shared abstract supertype are both named
# `TrajectorySampling` (Julia allows a type to share its enclosing module's
# name), but only one of the two may carry a docstring — Julia's docsystem
# can't disambiguate "the module" from "the same-named member inside it"
# when binding a docstring to the module itself, so this module has none;
# see the docstring on the type below instead.
module TrajectorySampling

import ..IntervalMDP:
    TrajectorySamplingStrategy,
    num_trajectories,
    terminate_sampling,
    action_selection,
    target_state_sampling,
    _omax_best_action,
    _categorical_sample,
    _action_uncertainty,
    _target_indices,
    _target_state

"""
    TrajectorySampling.TrajectorySampling <: TrajectorySamplingStrategy

Shared behaviour for this submodule's concrete strategies: one trajectory
per `sample` call, and an action selected greedily as
`argmax_a` of the O-max upper-bound Q-value at the current state. Concrete
subtypes need only implement `target_state_sampling`.
"""
abstract type TrajectorySampling <: TrajectorySamplingStrategy end

num_trajectories(::TrajectorySampling) = 1

# Stub for now — always continues until a reach/avoid state or the hard cap.
terminate_sampling(
    ::TrajectorySampling,
    current,
    trajectory,
    value_function,
    model,
    spec,
) = false

function action_selection(::TrajectorySampling, current, value_function, model, spec)
    a, _ = _omax_best_action(model, current, value_function.upper.current, true)
    return a
end

"""
    TransitionProbabilityTrajectorySampling()

Sample the next state directly from the O-max transition distribution
`p(·|s,a)`.
"""
struct TransitionProbabilityTrajectorySampling <: TrajectorySampling end

target_state_sampling(
    ::TransitionProbabilityTrajectorySampling,
    current,
    a,
    probs,
    value_function,
    model,
    spec,
) = _target_state(model, _categorical_sample(probs))

"""
    ExpectedGapTrajectorySampling()

Sample the next state with probability proportional to
`p(s'|s,a) * (U(s') - L(s'))` — biases toward successors whose value is
still uncertain.
"""
struct ExpectedGapTrajectorySampling <: TrajectorySampling end

function target_state_sampling(
    ::ExpectedGapTrajectorySampling,
    current,
    a,
    probs,
    value_function,
    model,
    spec,
)
    gap = value_function.upper.current .- value_function.lower.current
    return _target_state(model, _categorical_sample(probs .* vec(gap)))
end

"""
    ReachProbabilityTrajectorySampling()

Sample the next state with probability proportional to
`p(s'|s,a) * U(s')` — biases toward successors that look more likely to
reach the goal under the optimistic bound.
"""
struct ReachProbabilityTrajectorySampling <: TrajectorySampling end

function target_state_sampling(
    ::ReachProbabilityTrajectorySampling,
    current,
    a,
    probs,
    value_function,
    model,
    spec,
)
    return _target_state(
        model,
        _categorical_sample(probs .* vec(value_function.upper.current)),
    )
end

"""
    ActionUncertaintyTrajectorySampling()

Sample the next state with probability proportional to
`p(s'|s,a) * (U^{-a_L(s')}(s') - L(s'))`, where `a_L(s')` is the action
maximizing the lower-bound Q-value at `s'` and `U^{-a_L}(s')` is the best
upper-bound Q-value at `s'` among the *other* actions — biases toward
successors where it's still unclear whether the safe action is really
optimal.
"""
struct ActionUncertaintyTrajectorySampling <: TrajectorySampling end

function target_state_sampling(
    ::ActionUncertaintyTrajectorySampling,
    current,
    a,
    probs,
    value_function,
    model,
    spec,
)
    weighted = zeros(eltype(probs), length(probs))
    for sp in eachindex(probs)
        # Only score states the O-max realization actually assigned mass to
        # — multiplying by probs[sp] == 0 always contributes 0 regardless.
        probs[sp] > zero(eltype(probs)) || continue
        weighted[sp] =
            probs[sp] * _action_uncertainty(model, _target_state(model, sp), value_function)
    end
    return _target_state(model, _categorical_sample(weighted))
end

end # module TrajectorySampling

###################################
# 5. Priority-queue sampling       #
###################################

"""
    PriorityQueueSamplingStrategy <: SamplingStrategy

Abstract supertype for priority-based sampling strategies: states (or
`(action, state)` pairs) are ranked by some value-function-derived priority
(e.g. Bellman residual / gap) and the top-ranked entries are relaxed each
iteration. [`ValueFunctionOrderedSampling`](@ref) is the first, simplest
member — it recomputes and fully re-sorts the ranking from scratch every
iteration; a genuine priority queue that incrementally updates priorities
between iterations is a future addition to this category.

Concrete subtypes need the current value function and `Specification`, so
they implement `sample(strategy, model, strategy_cache, value_function,
spec)`.
"""
abstract type PriorityQueueSamplingStrategy <: SamplingStrategy end

sampling_context_requirement(::PriorityQueueSamplingStrategy) = NeedsValueFunctionAndSpec()

sample(ss::PriorityQueueSamplingStrategy, model, strategy_cache, value_function) =
    sample(ss, model, strategy_cache, value_function, nothing)

"""
    compute_priority(strategy, s, value_function, model, spec) -> Real

Strategy-specific priority for state `s` — higher means more urgent to
relax next. Must be implemented by every concrete strategy that
participates in incremental priority-queue sampling (see
[`PriorityQueueSampling.PriorityQueueSampling`](@ref)).
"""
function compute_priority end

"""
    ValueFunctionOrderedSampling(operation, ascending, k)

Value-function-ordered sampler: applies `operation` (e.g. `gap`) to the current
`IntervalValueFunction`, sorts states by the resulting values, and yields the
top `k` states. Yields bare states, so the inner loop of `bellman_update!`
sweeps all available actions per visited state.

# Fields
- `operation::Function`: maps an `IntervalValueFunction` to a per-state array.
- `ascending::Bool`: `true` for ascending (low-to-high), `false` for descending.
- `k::Int`: number of top states to select.
"""
struct ValueFunctionOrderedSampling <: PriorityQueueSamplingStrategy
    operation::Function  # e.g., gap; takes IntervalValueFunction, returns array
    ascending::Bool      # true for ascending (low-to-high), false for descending
    k::Int               # number of top states to select
end

struct ValueFunctionOrderedStateIterator{NS} <: AbstractIterator
    states::Vector{CartesianIndex{NS}}
end

Base.length(iter::ValueFunctionOrderedStateIterator) = length(iter.states)
Base.firstindex(iter::ValueFunctionOrderedStateIterator) = firstindex(iter.states)
Base.lastindex(iter::ValueFunctionOrderedStateIterator) = lastindex(iter.states)
Base.getindex(iter::ValueFunctionOrderedStateIterator, i) = iter.states[i]
Base.iterate(iter::ValueFunctionOrderedStateIterator) = iterate(iter.states)
Base.iterate(iter::ValueFunctionOrderedStateIterator, state) = iterate(iter.states, state)

sequence_shape(::ValueFunctionOrderedStateIterator) = StateUpdateSequence()

sample(ss::ValueFunctionOrderedSampling, model) =
    error("ValueFunctionOrderedSampling requires a value_function argument")
sample(ss::ValueFunctionOrderedSampling, model, strategy_cache) =
    error("ValueFunctionOrderedSampling requires a value_function argument")

function sample(ss::ValueFunctionOrderedSampling, model, strategy_cache, value_function, spec)
    return value_function_ordered_sample(ss.operation, ss.ascending, ss.k, value_function)
end

function value_function_ordered_sample(
    operation::Function,
    ascending::Bool,
    k::Int,
    value_function,
)
    # Apply the operation to get values
    values = operation(value_function)

    # Flatten to 1D and create index mapping
    flat_values = vec(values)
    indices = CartesianIndices(values)

    # Sort indices by values
    sorted_perm = if ascending
        sortperm(flat_values)
    else
        sortperm(flat_values; rev = true)
    end

    # Select top k indices
    k_selected = min(k, length(sorted_perm))
    selected_linear_indices = sorted_perm[1:k_selected]
    selected_states = [indices[i] for i in selected_linear_indices]

    return ValueFunctionOrderedStateIterator(selected_states)
end

###################################
# 5b. Priority-queue sampling      #
###################################
#
# Concrete incremental-priority-queue algorithms live in this submodule
# (`IntervalMDP.PriorityQueueSampling.GapPriorityQueueSampling()` etc.),
# mirroring `TrajectorySampling` (§4e): the general interface
# (`PriorityQueueSamplingStrategy`, `compute_priority`, `_successor_states`,
# `_state_indices`) stays in the parent `IntervalMDP` module, since
# `_gsrdp_sample`/`sampling_context_requirement` dispatch on it; only the
# concrete algorithms move into the submodule.
#
# Shared behaviour, all under `PriorityQueueSampling.PriorityQueueSampling`:
# the first `sample` call computes every state's priority from scratch; every
# later call only recomputes priorities for `_successor_states(model, s)` of
# each state `s` selected by the *previous* call (GSRDP always relaxes
# exactly what `sample` returns, so "states updated in the previous
# iteration" and "states `sample` returned last time" are the same set).
# Each call then returns the top `ss.k` states by priority.
#
#   * GapPriorityQueueSampling — priority(s) = U(s) - L(s).
#   * UpperBoundPriorityQueueSampling — priority(s) = U(s).
#   * ActionUncertaintyPriorityQueueSampling — priority(s) = V^a(s) (same
#     action-selection-uncertainty measure as
#     `TrajectorySampling.ActionUncertaintyTrajectorySampling`).
#
# NOTE: like `TrajectorySampling`, this module has no docstring of its own
# (a module can't carry one alongside a same-named member) — see the
# docstring on the type below instead.
module PriorityQueueSampling

import ..IntervalMDP:
    PriorityQueueSamplingStrategy,
    compute_priority,
    reset_sampling_strategy!,
    sample,
    StateIterator,
    _state_indices,
    _successor_states,
    _action_uncertainty

"""
    PriorityQueueSampling.PriorityQueueSampling <: PriorityQueueSamplingStrategy

Shared behaviour for this submodule's concrete strategies: a lazily
maintained priority over every state, initialized in full on the first
`sample` call and thereafter updated only where the previous call's
selections could have made it stale (`_successor_states` of each
previously-selected state), returning the top `k` states by priority each
call, breaking ties by least-recently-selected. Concrete subtypes need only
implement `compute_priority`.

Ties matter in practice, not just in theory: every non-goal/avoid state
starts with an identical priority under all three strategies below (`L = 0`,
`U = 1` everywhere before the first relaxation), so a naive tie-break that
always favors the same (e.g. lowest-index) state among ties would get stuck
reselecting it forever whenever `k` is smaller than the tied set — its own
priority never changes because nothing ever updates *its* predecessors, and
its own recompute (after being selected) leaves it tied with everyone else
again. Breaking ties by least-recently-selected guarantees the tied set
rotates instead of starving.
"""
abstract type PriorityQueueSampling <: PriorityQueueSamplingStrategy end

function reset_sampling_strategy!(ss::PriorityQueueSampling)
    ss.initialized[] = false
    ss.previous_selected[] = Int[]
    ss.clock[] = 0
    return nothing
end

function sample(ss::PriorityQueueSampling, model, strategy_cache, value_function, spec)
    S = _state_indices(model)
    nS = length(S)
    ss.clock[] += 1

    if !ss.initialized[]
        priorities = Vector{Float64}(undef, nS)
        @inbounds for i in 1:nS
            priorities[i] = compute_priority(ss, S[i], value_function, model, spec)
        end
        ss.priorities[] = priorities
        ss.last_selected[] = zeros(Int, nS)
        ss.initialized[] = true
    else
        priorities = ss.priorities[]
        stale = Set{eltype(S)}()
        for i in ss.previous_selected[]
            push!(stale, S[i])   # s's own priority is stale too — its value just changed
            union!(stale, _successor_states(model, S[i]))
        end
        L = LinearIndices(S)
        for sp in stale
            priorities[L[sp]] = compute_priority(ss, sp, value_function, model, spec)
        end
    end

    k = min(ss.k, nS)
    last_selected = ss.last_selected[]
    # Priorities within a small relative tolerance are treated as tied (not
    # just bit-identical ones): two states converging toward the same
    # asymptotic priority from a shared, still-stale neighborhood can settle
    # into a persistent floating-point-scale gap (e.g. 1e-9 relative) that a
    # strict `<` comparison would treat as a real, permanent difference —
    # starving whichever state loses it forever, even though both still need
    # further relaxation. Rounding to a coarse relative precision before
    # comparing lets the recency tie-break take over once that happens,
    # without disturbing genuine (much larger) priority differences.
    key = i -> (-round(priorities[i]; sigdigits = 6), last_selected[i])
    top = partialsortperm(1:nS, 1:k; by = key)
    for i in top
        last_selected[i] = ss.clock[]
    end
    ss.previous_selected[] = top
    return StateIterator([S[i] for i in top])
end

"""
    GapPriorityQueueSampling(k)

Priority-queue sampler: `priority(s) = U(s) - L(s)`, the value-function
gap — biases toward states whose bounds are still furthest apart. Yields
the top `k` states by priority each call.
"""
struct GapPriorityQueueSampling <: PriorityQueueSampling
    k::Int
    priorities::Base.RefValue{Vector{Float64}}
    last_selected::Base.RefValue{Vector{Int}}
    clock::Base.RefValue{Int}
    initialized::Base.RefValue{Bool}
    previous_selected::Base.RefValue{Vector{Int}}

    GapPriorityQueueSampling(k::Int) =
        new(k, Ref(Float64[]), Ref(Int[]), Ref(0), Ref(false), Ref(Int[]))
end

compute_priority(::GapPriorityQueueSampling, s, value_function, model, spec) =
    value_function.upper.current[s] - value_function.lower.current[s]

"""
    UpperBoundPriorityQueueSampling(k)

Priority-queue sampler: `priority(s) = U(s)`, the upper (optimistic) value —
biases toward states that look most valuable under the optimistic bound.
Yields the top `k` states by priority each call.
"""
struct UpperBoundPriorityQueueSampling <: PriorityQueueSampling
    k::Int
    priorities::Base.RefValue{Vector{Float64}}
    last_selected::Base.RefValue{Vector{Int}}
    clock::Base.RefValue{Int}
    initialized::Base.RefValue{Bool}
    previous_selected::Base.RefValue{Vector{Int}}

    UpperBoundPriorityQueueSampling(k::Int) =
        new(k, Ref(Float64[]), Ref(Int[]), Ref(0), Ref(false), Ref(Int[]))
end

compute_priority(::UpperBoundPriorityQueueSampling, s, value_function, model, spec) =
    value_function.upper.current[s]

"""
    ActionUncertaintyPriorityQueueSampling(k)

Priority-queue sampler: `priority(s) = V^a(s)`, the same action-selection
uncertainty measure
[`TrajectorySampling.ActionUncertaintyTrajectorySampling`](@ref) weights
by — `U^{-a_L(s)}(s) - L(s)`, where `a_L(s)` is the action maximizing the
lower-bound Q-value at `s` and `U^{-a_L}(s)` is the best upper-bound
Q-value at `s` among the *other* actions. Yields the top `k` states by
priority each call.
"""
struct ActionUncertaintyPriorityQueueSampling <: PriorityQueueSampling
    k::Int
    priorities::Base.RefValue{Vector{Float64}}
    last_selected::Base.RefValue{Vector{Int}}
    clock::Base.RefValue{Int}
    initialized::Base.RefValue{Bool}
    previous_selected::Base.RefValue{Vector{Int}}

    ActionUncertaintyPriorityQueueSampling(k::Int) =
        new(k, Ref(Float64[]), Ref(Int[]), Ref(0), Ref(false), Ref(Int[]))
end

compute_priority(::ActionUncertaintyPriorityQueueSampling, s, value_function, model, spec) =
    _action_uncertainty(model, s, value_function)

end # module PriorityQueueSampling

###################################
# 6. Given sequence                #
###################################

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

###################################
# 7. Composite sampling            #
###################################

"""
    CompositeSamplingStrategy <: SamplingStrategy

Abstract supertype for sampling strategies that combine one or more other
`SamplingStrategy` instances rather than sampling states directly
themselves — e.g. mixing two strategies, or filtering down another
strategy's output. Because a wrapped sub-strategy may itself need the value
function and/or `Specification` (a [`TrajectorySamplingStrategy`](@ref) or
[`PriorityQueueSamplingStrategy`](@ref)), composites always receive the
richest signature (`sample(strategy, model, strategy_cache, value_function,
spec)`) and forward to each sub-strategy via `_gsrdp_sample`, which
dispatches on that sub-strategy's own `sampling_context_requirement`.

Concrete composites must implement `sub_strategies(strategy)`, returning a
tuple of the wrapped strategies, so `reset_sampling_strategy!` propagates to
them automatically.
"""
abstract type CompositeSamplingStrategy <: SamplingStrategy end

sampling_context_requirement(::CompositeSamplingStrategy) = NeedsValueFunctionAndSpec()

sample(ss::CompositeSamplingStrategy, model, strategy_cache, value_function) =
    sample(ss, model, strategy_cache, value_function, nothing)

"""
    RandomlyThinned(base, keep_prob)

Wraps `base`, independently keeping each entry of its update sequence with
probability `keep_prob` (dropping it otherwise). Turns any strategy "leaky" —
e.g. `RandomlyThinned(RoundRobinState(k), 0.8)` round-robins through states
but only actually relaxes about 80% of the scheduled window each iteration.
"""
struct RandomlyThinned{S <: SamplingStrategy} <: CompositeSamplingStrategy
    base::S
    keep_prob::Float64

    function RandomlyThinned(base::S, keep_prob::Real) where {S <: SamplingStrategy}
        0 <= keep_prob <= 1 ||
            throw(ArgumentError("keep_prob must be in [0, 1], got $keep_prob"))
        return new{S}(base, Float64(keep_prob))
    end
end

sub_strategies(ss::RandomlyThinned) = (ss.base,)

function sample(ss::RandomlyThinned, model, strategy_cache, value_function, spec)
    seq = _gsrdp_sample(ss.base, model, strategy_cache, value_function, spec)
    return _thin(sequence_shape(seq), seq, model, ss.keep_prob)
end

# `seq`'s elements aren't materialized via a plain comprehension here: these
# custom iterators don't define `Base.eltype`, so a comprehension would
# infer `Vector{Any}` — which can't convert into `RandomSubsetState(Action)?Iterator`'s
# concretely-typed field. Pre-typing `kept` from `model`'s own shapes (the
# same source `ss.base`'s `sample` used) keeps this concrete.
function _thin(::StateUpdateSequence, seq, model, keep_prob::Float64)
    S = CartesianIndices(source_shape(model))
    kept = Vector{eltype(S)}(undef, 0)
    for s in seq
        rand() < keep_prob && push!(kept, s)
    end
    return RandomSubsetStateIterator(kept)
end

function _thin(::StateActionUpdateSequence, seq, model, keep_prob::Float64)
    A = CartesianIndices(action_shape(model))
    S = CartesianIndices(source_shape(model))
    kept = Vector{Tuple{eltype(A), eltype(S)}}(undef, 0)
    for x in seq
        rand() < keep_prob && push!(kept, x)
    end
    return RandomSubsetStateActionIterator(kept)
end

"""
    EpsilonGreedyMixture(exploit, explore, epsilon)

Each iteration, samples from `explore` with probability `epsilon` and from
`exploit` otherwise. Lets sampling mix two different strategies — e.g.
mostly follow a priority-queue or trajectory strategy (`exploit`) but
occasionally fall back to uniform random exploration (`explore =
RandomSubsetState(k)`) to avoid starving states `exploit` never visits.
"""
struct EpsilonGreedyMixture{E1 <: SamplingStrategy, E2 <: SamplingStrategy} <:
       CompositeSamplingStrategy
    exploit::E1
    explore::E2
    epsilon::Float64

    function EpsilonGreedyMixture(
        exploit::E1,
        explore::E2,
        epsilon::Real,
    ) where {E1 <: SamplingStrategy, E2 <: SamplingStrategy}
        0 <= epsilon <= 1 || throw(ArgumentError("epsilon must be in [0, 1], got $epsilon"))
        return new{E1, E2}(exploit, explore, Float64(epsilon))
    end
end

sub_strategies(ss::EpsilonGreedyMixture) = (ss.exploit, ss.explore)

function sample(ss::EpsilonGreedyMixture, model, strategy_cache, value_function, spec)
    chosen = rand() < ss.epsilon ? ss.explore : ss.exploit
    return _gsrdp_sample(chosen, model, strategy_cache, value_function, spec)
end
