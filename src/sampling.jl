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
# `getindex` takes a 1-based linear index with the actions as the fast axis
# (stride `nA`), so the index bounds are those of `1:nA*nS` — independent of
# the axes of `A` and `S`, whose offsets `getindex` adds itself.
Base.firstindex(iter::ProductIterator) = 1
Base.lastindex(iter::ProductIterator) = iter.nA * iter.nS
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

    return (A[firstindex(A) - 1 + i], S[firstindex(S) - 1 + i])
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

random_subset_state_action_sample(k::Int, proc::ProductProcess) =
    random_subset_state_action_sample(k, markov_process(proc))

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
sample(ss::RoundRobinState, model, ::AbstractStrategyCache) =
    round_robin_state_sample(ss, model)

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
sampling states independently, these simulate a trajectory through the model —
starting from an initial state, repeatedly selecting an action, realizing a
concrete transition distribution out of the ambiguity set via O-maximization,
and drawing a next state from it — and relax the states visited along the way.

The category lives here, alongside the O-max primitives every trajectory step
is built from (§4a below), because `sampling_context_requirement` dispatches on
it and the priority-queue strategies share those primitives. The sampler itself
is in `trajectorysampling.jl`: a single configurable
[`TrajectorySampling.TrajectorySampling`](@ref), parameterized by a selection
policy, action and successor score functions, the concrete-transition bound and
adversary direction, and Gauss-Seidel batching.

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
            "trajectory and priority-queue sampling only support flat (non-factored) " *
            "models; got $(length(ms)) marginals",
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
    _omax_best_action(model, jₛ, V, upper_bound; exclude=nothing, maximize=true) -> (action_or_nothing, value_or_nothing)

The available action at state `jₛ` maximizing [`_omax_expectation`](@ref)
against `V`/`upper_bound` — i.e. `argmax_a Q(jₛ, a)` under the O-max
Q-value for that direction, or `argmin_a` with `maximize = false`, matching
a `Minimize` specification. With `exclude`, optimizes among every available
action *except* `exclude`. Returns `(nothing, nothing)` if no candidate
action exists (no actions available at `jₛ`, or `exclude` was the only one).
"""
function _omax_best_action(
    model,
    jₛ,
    V,
    upper_bound::Bool;
    exclude = nothing,
    maximize::Bool = true,
)
    marginal = _omax_marginal(model)
    best_a, best_v = nothing, nothing
    for jₐ in available(model, jₛ)
        (exclude !== nothing && jₐ == exclude) && continue
        v = _omax_expectation(marginal[jₐ, jₛ], V, upper_bound)
        if best_v === nothing || (maximize ? v > best_v : v < best_v)
            best_a, best_v = jₐ, v
        end
    end
    return best_a, best_v
end

"""
    _action_uncertainty(model, sp, value_function, spec) -> Real

`V^a(s') = U^{-a_L(s')}(s') - L(s')`, where `a_L(s')` is the action
maximizing the lower-bound Q-value at `s'` and `U^{-a_L}(s')` is the best
upper-bound Q-value at `s'` among the *other* actions — a measure of how
uncertain it still is whether the (lower-bound-)optimal action at `s'`
really is optimal. Returns 0 when `s'` has only one available action (no
alternative to be uncertain about). Used by
[`PriorityQueueSampling.ActionUncertaintyPriorityQueueSampling`](@ref).

Both `_omax_best_action` calls use the *same* `isoptimistic(spec)` direction: `L` and `U` are
themselves both derived from `bellman_update!` backups under that one direction — their bracket
comes from the strategy cache (optimizing vs following), not from opposing O-max/O-min
directions — so re-deriving each one's own best action must match that same direction.
"""
function _action_uncertainty(model, sp, value_function, spec)
    L, U = value_function.lower.current, value_function.upper.current
    # An implicit sink (a target state outside the source sub-box) has no
    # ambiguity-set column and no actions at all, so there is no action to be
    # uncertain about — same reasoning as the single-action case below.
    _is_source_state(model, sp) || return zero(eltype(U))
    dir = _isoptimistic(spec)
    a_L, L_sp = _omax_best_action(model, sp, L, dir)
    _, U_excl = _omax_best_action(model, sp, U, dir; exclude = a_L)
    U_excl === nothing && return zero(eltype(U))   # only one action at sp: no alternative to be uncertain about
    return U_excl - L_sp
end

# Nothing-tolerant `isoptimistic`, mirroring `_delta_maximize` below (§5c) — some unit tests
# exercise these O-max helpers directly without a full `Specification`.
_isoptimistic(::Nothing) = true
_isoptimistic(spec) = isoptimistic(spec)

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
`_categorical_sample` (`trajectorysampling.jl`), or from an ambiguity set's
`support`, maps back to a state index. Only equals `CartesianIndex(i)` for single-state-variable
models.
"""
Base.@propagate_inbounds _target_state(model, i) = _target_indices(model)[i]

"""
    _maybe_target_state(model, i) -> Union{CartesianIndex, Nothing}

[`_target_state`](@ref) that passes `nothing` straight through — the sentinel
`_categorical_sample` (`trajectorysampling.jl`) returns when a weight vector
carries no positive mass, which the trajectory rollout forwards as "no
successor to move to", ending the trajectory.
"""
Base.@propagate_inbounds _maybe_target_state(model, i::Integer) = _target_state(model, i)
_maybe_target_state(model, ::Nothing) = nothing

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
    _predecessor_states(model, s) -> Set

`Pred(s) = {s' | ∃a. p̄(s|s',a) > 0}` — every source state that can reach `s`
in one step under some available action. Used by incremental priority-queue
sampling to find which states' priorities may have gone stale after `s` was
relaxed: a change in `V(s)` propagates *backward*, to the states whose
Bellman backup reads `V(s)`.

Membership is decided on the *upper* transition probability being nonzero,
not on `support`: `support` returns the full target range for a dense
`IntervalAmbiguitySets` (see its docstring), so going by support alone would
report every state as a predecessor of every other.

Implicit sink states (see [`_is_source_state`](@ref)) can never be
predecessors — they own no ambiguity-set column and no actions — so no
filtering is needed here, unlike in the forward direction. A sink may still
be passed as `s`; it simply has no outgoing edges of its own.

O(|S|·|A|) per query. [`_predecessor_index`](@ref) computes the whole
relation in a single pass, for callers that need it repeatedly.
"""
function _predecessor_states(model, s)
    marginal = _omax_marginal(model)
    i = LinearIndices(_target_indices(model))[s]

    S = _state_indices(model)
    pred = Set{eltype(S)}()
    for sp in S
        any(a -> upper(marginal[a, sp], i) > 0, available(model, sp)) && push!(pred, sp)
    end
    return pred
end

"""
    _predecessor_index(model) -> Vector{Vector{Tuple{Int, Float64}}}

The full predecessor relation, computed in one pass over every
`(source, action)` ambiguity set: entry `i` — a *target* linear index —
lists `(source linear index, maxₐ p̄(target | source, a))` for every
predecessor of that target. Only `maxₐ p̄` is retained, since prioritised
sweeping propagates `δ_pred = maxₐ p̄(s | s_pred, a) · |Δ(s)|`.

Source indices are linear into [`_state_indices`](@ref), target indices
linear into [`_target_indices`](@ref) — the two ranges differ whenever the
model declares implicit sink states.

Callers cache the result across `sample` calls, which is sound because GSRDP
is infinite-horizon (hence a stationary model) and `_omax_marginal` already
rejects anything but a flat, single-marginal model.
"""
function _predecessor_index(model)
    marginal = _omax_marginal(model)
    S = _state_indices(model)
    L = LinearIndices(S)

    index = [Tuple{Int, Float64}[] for _ in 1:length(_target_indices(model))]
    # `maxₐ p̄` per target, accumulated over the actions available at one source.
    acc = Dict{Int, Float64}()

    for sp in S
        empty!(acc)
        for a in available(model, sp)
            ambiguity_set = marginal[a, sp]
            for i in support(ambiguity_set)
                p = upper(ambiguity_set, i)
                p > 0 || continue
                acc[i] = max(get(acc, i, zero(Float64)), Float64(p))
            end
        end

        lsp = L[sp]
        for (i, p) in acc
            push!(index[i], (lsp, p))
        end
    end

    return index
end

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

function sample(
    ss::ValueFunctionOrderedSampling,
    model,
    strategy_cache,
    value_function,
    spec,
)
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
# mirroring `TrajectorySampling` (`trajectorysampling.jl`): the general interface
# (`PriorityQueueSamplingStrategy`, `compute_priority`, `_predecessor_states`,
# `_state_indices`) stays in the parent `IntervalMDP` module, since
# `_gsrdp_sample`/`sampling_context_requirement` dispatch on it; only the
# concrete algorithms move into the submodule.
#
# Shared behaviour, all under `PriorityQueueSampling.PriorityQueueSampling`:
# the first `sample` call computes every state's priority from scratch; every
# later call only recomputes priorities for the *predecessors*
# (`_predecessor_index`) of each state `s` selected by the *previous* call —
# prioritised sweeping's backward propagation, since a change in `V(s)` is
# read by exactly the states that can transition into `s`. (GSRDP always
# relaxes exactly what `sample` returns, so "states updated in the previous
# iteration" and "states `sample` returned last time" are the same set.)
# Each call then returns the top `ss.k` states by priority.
#
#   * GapPriorityQueueSampling — priority(s) = U(s) - L(s).
#   * UpperBoundPriorityQueueSampling — priority(s) = U(s).
#   * ActionUncertaintyPriorityQueueSampling — priority(s) = V^a(s), the
#     action-selection-uncertainty measure `_action_uncertainty` computes.
#   * RNDPriorityQueueSampling — priority(s) = max(δ(s), λ · novelty(s)),
#     with novelty from a Random Network Distillation predictor standing in
#     for a per-state backup counter (§5c).
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
    _target_indices,
    _predecessor_index,
    _action_uncertainty,
    _omax_best_action,
    _is_source_state,
    ismaximize,
    _isoptimistic,
    _rnd_construct,
    _rnd_novelty,
    _rnd_train!,
    _rnd_calibrate!

"""
    PriorityQueueSampling.PriorityQueueSampling <: PriorityQueueSamplingStrategy

Shared behaviour for this submodule's concrete strategies: a lazily
maintained priority over every state, initialized in full on the first
`sample` call and thereafter updated only where the previous call's
selections could have made it stale (the *predecessors* of each
previously-selected state, via `_predecessor_index`), returning the top `k`
states by priority each call, breaking ties by least-recently-selected.
Concrete subtypes need only implement `compute_priority`; the optional hooks
[`propagated_priority`](@ref), [`on_selected!`](@ref) and
[`initialize_priority_state!`](@ref) default to behaviour that ignores
prioritised sweeping's propagated magnitude.

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
    ss.selected_snapshot[] = Tuple{Float64, Float64}[]
    ss.predecessor_index[] = Vector{Tuple{Int, Float64}}[]
    ss.clock[] = 0
    return nothing
end

"""
    propagated_priority(strategy, sp, propagated, value_function, model, spec) -> Real

Priority for a state `sp` whose priority went stale because a *successor* of
it was relaxed. `propagated` is `maxₐ p̄(s | sp, a) · |Δ(s)|` maximised over
the relaxed states `s` that `sp` can reach — the bound on how much `sp`'s own
value can move as a result.

Defaults to [`compute_priority`](@ref), i.e. ignoring the propagated
magnitude and simply recomputing the strategy's own priority, which is what
the value-function-derived strategies here want. Strategies that combine the
two — [`RNDPriorityQueueSampling`](@ref) — override it.
"""
propagated_priority(
    ss::PriorityQueueSampling,
    sp,
    propagated,
    value_function,
    model,
    spec,
) = compute_priority(ss, sp, value_function, model, spec)

"""
    on_selected!(strategy, states, value_function, model, spec)

Hook called with the states a `sample` call selected, immediately after
selection and hence immediately before GSRDP relaxes exactly those states.
Defaults to a no-op; [`RNDPriorityQueueSampling`](@ref) uses it to train its
novelty predictor on the states that are about to be backed up.
"""
on_selected!(::PriorityQueueSampling, states, value_function, model, spec) = nothing

# The value pair snapshotted for a selected state, so that the *actual*
# backup magnitude |Δ(s)| can be measured on the next call. It has to be
# measured rather than read off the value function: `_gsrdp!` calls
# `nextiteration!` (which copies `current` into `previous`) immediately
# before `sample`, so by the time a strategy is asked for a sample the two
# carry identical values and no residual.
_snapshot(value_function, s) =
    (Float64(value_function.upper.current[s]), Float64(value_function.lower.current[s]))

_backup_magnitude(value_function, s, snapshot) = max(
    abs(Float64(value_function.upper.current[s]) - snapshot[1]),
    abs(Float64(value_function.lower.current[s]) - snapshot[2]),
)

function sample(ss::PriorityQueueSampling, model, strategy_cache, value_function, spec)
    S = _state_indices(model)
    nS = length(S)
    ss.clock[] += 1

    if !ss.initialized[]
        ss.predecessor_index[] = _predecessor_index(model)
        initialize_priority_state!(ss, S, value_function, model, spec)

        # Assigned before the sweep, not after: strategies whose priority
        # depends on whether a state has been relaxed yet read it from here.
        ss.last_selected[] = zeros(Int, nS)

        priorities = Vector{Float64}(undef, nS)
        @inbounds for i in 1:nS
            priorities[i] = compute_priority(ss, S[i], value_function, model, spec)
        end
        ss.priorities[] = priorities
        ss.initialized[] = true
    else
        priorities = ss.priorities[]
        index = ss.predecessor_index[]
        target_linear = LinearIndices(_target_indices(model))

        # Backward propagation: relaxing `s` can only move the value of a
        # state that reads `V(s)`, i.e. a predecessor of `s`, and by at most
        # `maxₐ p̄(s | sp, a) · |Δ(s)|`.
        propagated = Dict{Int, Float64}()
        snapshots = ss.selected_snapshot[]
        for (n, i) in enumerate(ss.previous_selected[])
            Δ = _backup_magnitude(value_function, S[i], snapshots[n])
            for (j, p̄) in index[target_linear[S[i]]]
                propagated[j] = max(get(propagated, j, 0.0), p̄ * Δ)
            end
        end

        for (j, m) in propagated
            priorities[j] = propagated_priority(ss, S[j], m, value_function, model, spec)
        end

        # A relaxed state's own priority is stale too — its value just
        # changed. If it is also its own predecessor (a self-loop), keep
        # whichever of the two readings is more urgent.
        for i in ss.previous_selected[]
            own = compute_priority(ss, S[i], value_function, model, spec)
            priorities[i] = haskey(propagated, i) ? max(priorities[i], own) : own
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
    ss.selected_snapshot[] = [_snapshot(value_function, S[i]) for i in top]

    selected = [S[i] for i in top]
    on_selected!(ss, selected, value_function, model, spec)

    return StateIterator(selected)
end

"""
    initialize_priority_state!(strategy, S, value_function, model, spec)

Hook called once per `solve`, before the initial full priority sweep, with
every state `S`. Defaults to a no-op; strategies carrying a lazily
constructed model — [`RNDPriorityQueueSampling`](@ref) and its novelty
networks, whose input dimension is only known once a model is in hand — build
it here so that the sweep that follows can already query it.
"""
initialize_priority_state!(::PriorityQueueSampling, S, value_function, model, spec) =
    nothing

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
    selected_snapshot::Base.RefValue{Vector{Tuple{Float64, Float64}}}
    predecessor_index::Base.RefValue{Vector{Vector{Tuple{Int, Float64}}}}

    GapPriorityQueueSampling(k::Int) = new(
        k,
        Ref(Float64[]),
        Ref(Int[]),
        Ref(0),
        Ref(false),
        Ref(Int[]),
        Ref(Tuple{Float64, Float64}[]),
        Ref(Vector{Tuple{Int, Float64}}[]),
    )
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
    selected_snapshot::Base.RefValue{Vector{Tuple{Float64, Float64}}}
    predecessor_index::Base.RefValue{Vector{Vector{Tuple{Int, Float64}}}}

    UpperBoundPriorityQueueSampling(k::Int) = new(
        k,
        Ref(Float64[]),
        Ref(Int[]),
        Ref(0),
        Ref(false),
        Ref(Int[]),
        Ref(Tuple{Float64, Float64}[]),
        Ref(Vector{Tuple{Int, Float64}}[]),
    )
end

compute_priority(::UpperBoundPriorityQueueSampling, s, value_function, model, spec) =
    value_function.upper.current[s]

"""
    ActionUncertaintyPriorityQueueSampling(k)

Priority-queue sampler: `priority(s) = V^a(s)`, the action-selection
uncertainty measure `_action_uncertainty` computes —
`U^{-a_L(s)}(s) - L(s)`, where `a_L(s)` is the action maximizing the
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
    selected_snapshot::Base.RefValue{Vector{Tuple{Float64, Float64}}}
    predecessor_index::Base.RefValue{Vector{Vector{Tuple{Int, Float64}}}}

    ActionUncertaintyPriorityQueueSampling(k::Int) = new(
        k,
        Ref(Float64[]),
        Ref(Int[]),
        Ref(0),
        Ref(false),
        Ref(Int[]),
        Ref(Tuple{Float64, Float64}[]),
        Ref(Vector{Tuple{Int, Float64}}[]),
    )
end

compute_priority(::ActionUncertaintyPriorityQueueSampling, s, value_function, model, spec) =
    _action_uncertainty(model, s, value_function, spec)

###################################
# 5c. RND-based priority queue     #
###################################
#
# `δ` is deliberately pluggable — the three functions below are ready-made,
# and any `(s, value_function, model, spec) -> Real` works.

"""
    bellman_residual_delta(s, value_function, model, spec) -> Real

`δ(s) = |maxₐ Q_U(s, a) − U(s)|`, the Bellman residual at `s` under the
upper bound: how far `s` still is from satisfying its own Bellman equation,
and hence how much a backup there would move it. The default `δ` for
[`RNDPriorityQueueSampling`](@ref).

Measured on the upper bound because that is the bound GSRDP lets drive
action selection; both the O-max/O-min ambiguity-set direction and the
maximize/minimize action-selection direction follow the specification
(`isoptimistic`/`Maximize`-`Minimize` respectively), defaulting to
optimistic-and-maximizing when `spec === nothing`. The direction must match
`bellman_update!`'s for `U` — this is a residual against `U`'s own equation,
so re-deriving it with a different direction would check the wrong equation.
"""
function bellman_residual_delta(s, value_function, model, spec)
    _is_source_state(model, s) || return 0.0

    U = value_function.upper.current
    _, q = _omax_best_action(
        model,
        s,
        U,
        _isoptimistic(spec);
        maximize = _delta_maximize(spec),
    )
    isnothing(q) && return 0.0

    return abs(Float64(q) - Float64(U[s]))
end

_delta_maximize(::Nothing) = true
_delta_maximize(spec) = ismaximize(spec)

"""
    gap_delta(s, value_function, model, spec) -> Real

`δ(s) = |U(s) − L(s)|`, the value-function gap — the same measure
[`GapPriorityQueueSampling`](@ref) prioritizes by, usable as a `δ` for
[`RNDPriorityQueueSampling`](@ref).
"""
gap_delta(s, value_function, model, spec) =
    abs(Float64(value_function.upper.current[s]) - Float64(value_function.lower.current[s]))

"""
    action_uncertainty_delta(s, value_function, model, spec) -> Real

`δ(s) = V^a(s)`, the action-selection uncertainty
[`ActionUncertaintyPriorityQueueSampling`](@ref) prioritizes by, usable as a
`δ` for [`RNDPriorityQueueSampling`](@ref).
"""
action_uncertainty_delta(s, value_function, model, spec) =
    _action_uncertainty(model, s, value_function, spec)

"""
    RNDPriorityQueueSampling(k; delta, lambda, hidden, output, lr, epochs, features, rng)

Priority-queue sampler whose priority is a *floor* combination of a
value-function residual and a Random Network Distillation novelty signal:

    priority(s) = max(δ(s), λ · novelty(s))

The model is known exactly, so `δ(s)` is trustworthy on its own and novelty
must not discount it — which is why this is a floor rather than the
multiplicative form RND takes under model uncertainty. What the floor buys
is that a region which has never been backed up still gets swept while its
`δ` is artificially small from a cold start.

`novelty(s)` is not "have I seen data here" — the model is known everywhere,
so that question is empty. It is a *generalized backup-recency* signal: the
predictor is trained on every state that gets backed up, so its error decays
across the whole neighbourhood of well-swept regions and stays high
elsewhere. That makes it a function-approximated stand-in for a per-state
backup counter, which a state space too large to enumerate cannot maintain
exactly. See [`RandomNetworkDistillation`](@ref).

Train vs. evaluate follows from a backup being the grounded event here:
a state that is popped and relaxed is trained on; a predecessor being
assigned a propagated priority, and the initial seeding sweep, only
evaluate. The floor is applied at exactly those evaluation points — that is,
to states that have not been relaxed yet — and drops away once a state has
been swept; see [`_novelty_floor`](@ref), which also explains why that is
what keeps the sampler live.

Propagation is prioritised sweeping's: after `s` is relaxed by `Δ(s)`, each
predecessor `s_pred` is pushed with `max(maxₐ p̄(s|s_pred,a) · |Δ(s)|,
λ · novelty(s_pred))`.

# Keywords
- `delta`: `(s, value_function, model, spec) -> Real`, the `δ` above.
  Defaults to [`bellman_residual_delta`](@ref); [`gap_delta`](@ref) and
  [`action_uncertainty_delta`](@ref) are also provided.
- `lambda`: the novelty floor's weight. Novelty is normalized to start near
  `1`, so `λ` is in the same units as `δ`. `λ = 0` disables the floor and
  reduces this to plain prioritised sweeping over `δ`.
- `hidden`, `output`, `lr`, `epochs`: novelty network size, Adam learning
  rate, and gradient steps per relaxed batch.
- `features`: `s::CartesianIndex -> Vector{Float32}` state embedding.
  Defaults to normalized state-variable indices; models with a meaningful
  geometry should pass their own, since novelty generalizes exactly as far
  as the embedding says two states are alike.
- `rng`: optional RNG for network initialization, for reproducibility.
"""
struct RNDPriorityQueueSampling <: PriorityQueueSampling
    k::Int
    delta::Function
    lambda::Float64
    hidden::Int
    output::Int
    lr::Float64
    epochs::Int
    features::Any
    rng::Any
    # Built lazily in `initialize_priority_state!`: the networks' input
    # dimension is only known once there is a model to read a state's
    # features from.
    rnd::Base.RefValue{Any}
    priorities::Base.RefValue{Vector{Float64}}
    last_selected::Base.RefValue{Vector{Int}}
    clock::Base.RefValue{Int}
    initialized::Base.RefValue{Bool}
    previous_selected::Base.RefValue{Vector{Int}}
    selected_snapshot::Base.RefValue{Vector{Tuple{Float64, Float64}}}
    predecessor_index::Base.RefValue{Vector{Vector{Tuple{Int, Float64}}}}

    function RNDPriorityQueueSampling(
        k::Int;
        delta::Function = bellman_residual_delta,
        lambda::Real = 1.0,
        hidden::Int = 32,
        output::Int = 8,
        lr::Real = 1e-3,
        epochs::Int = 1,
        features = nothing,
        rng = nothing,
    )
        lambda >= 0 || throw(ArgumentError("lambda must be non-negative, got $lambda"))
        epochs >= 1 || throw(ArgumentError("epochs must be positive, got $epochs"))

        return new(
            k,
            delta,
            Float64(lambda),
            hidden,
            output,
            Float64(lr),
            epochs,
            features,
            rng,
            Ref{Any}(nothing),
            Ref(Float64[]),
            Ref(Int[]),
            Ref(0),
            Ref(false),
            Ref(Int[]),
            Ref(Tuple{Float64, Float64}[]),
            Ref(Vector{Tuple{Int, Float64}}[]),
        )
    end
end

function reset_sampling_strategy!(ss::RNDPriorityQueueSampling)
    invoke(reset_sampling_strategy!, Tuple{PriorityQueueSampling}, ss)
    # Fresh networks per `solve`: a predictor still carrying the previous
    # run's training would report a whole region as already well-swept
    # before this run has backed up anything at all.
    ss.rnd[] = nothing
    return nothing
end

function initialize_priority_state!(
    ss::RNDPriorityQueueSampling,
    S,
    value_function,
    model,
    spec,
)
    ss.rnd[] = _rnd_construct(
        model;
        hidden = ss.hidden,
        output = ss.output,
        lr = ss.lr,
        epochs = ss.epochs,
        features = ss.features,
        rng = ss.rng,
    )
    # Calibrate before any training, so novelty starts near 1 everywhere and
    # `λ` reads as "the priority floor a never-backed-up state gets".
    _rnd_calibrate!(ss.rnd[], S)

    return nothing
end

"""
    _relaxed(strategy, s, model) -> Bool

Whether `s` has been selected — and hence relaxed by GSRDP — at least once
this run, read off the tie-break clock.
"""
function _relaxed(ss::RNDPriorityQueueSampling, s, model)
    last_selected = ss.last_selected[]
    isempty(last_selected) && return false
    return last_selected[LinearIndices(_state_indices(model))[s]] > 0
end

"""
    _novelty_floor(strategy, s, model) -> Real

`λ · novelty(s)`, but only for a state that has not been relaxed yet; `0`
once it has.

This is what the floor is *for*: novelty is evaluated when a state enters
the queue — at seeding, and when a predecessor is pushed — to keep a region
that has never been swept from being ignored while its `δ` is still
artificially small. A state that has already been backed up is no longer
making that claim, and re-applying the floor to it would be reading the
predictor as a statement about the *future* rather than about coverage so
far.

It is also what makes the sampler live. The novelty ordering across states
is arbitrary — it comes from a random target network — and training
generalizes, so every state's novelty decays roughly together. A floor that
applied forever would therefore freeze that arbitrary ordering into the
priorities: with `k` smaller than the number of states, the states that
happen to rank lowest would never be selected, and their bounds would never
converge. Restricting the floor to never-relaxed states means every
priority eventually falls back to `δ` and propagation, so an unswept state
is guaranteed to reach the top of the queue once the swept ones converge.
"""
function _novelty_floor(ss::RNDPriorityQueueSampling, s, model)
    (iszero(ss.lambda) || _relaxed(ss, s, model)) && return 0.0

    rnd = ss.rnd[]
    isnothing(rnd) && throw(
        ArgumentError(
            "the novelty networks have not been built yet — `sample` builds them on " *
            "its first call, so `compute_priority` is only meaningful after that",
        ),
    )

    return ss.lambda * _rnd_novelty(rnd, s)
end

compute_priority(ss::RNDPriorityQueueSampling, s, value_function, model, spec) =
    max(Float64(ss.delta(s, value_function, model, spec)), _novelty_floor(ss, s, model))

propagated_priority(
    ss::RNDPriorityQueueSampling,
    sp,
    propagated,
    value_function,
    model,
    spec,
) = max(Float64(propagated), _novelty_floor(ss, sp, model))

# A backup is the grounded event this signal tracks, so training happens on
# exactly the states GSRDP is about to relax — and nowhere else.
on_selected!(ss::RNDPriorityQueueSampling, states, value_function, model, spec) =
    (_rnd_train!(ss.rnd[], states); nothing)

end # module PriorityQueueSampling

###################################
# 6. Given sequence                #
###################################

"""
    GivenSequence()

Replays a caller-supplied update sequence verbatim: `sample(GivenSequence(),
model, sequence)` takes a `Vector` of `(action_tuple, state_tuple)` pairs and
yields exactly those, in order, after bounds-checking them against the model's
shapes. Primarily for tests and for reproducing a recorded sweep.
"""
struct GivenSequence <: SamplingStrategy end

function sample(
    ::GivenSequence,
    model,
    sequence::Vector{Tuple{NTuple{NA, T}, NTuple{NS, T}}}, # each element: (action_tuple, state_tuple)
) where {NA, NS, T <: Integer}
    return custom_sequence(model, sequence)
end

# Elements are `(action_tuple, state_tuple)`, matching the `(action, state)`
# order every other update sequence yields (see `ProductIterator`) and the
# order `GivenSequenceIterator` destructures them in.
function custom_sequence(
    model::FactoredRMDP,
    sequence::Vector{Tuple{NTuple{NA, T}, NTuple{NS, T}}},
) where {NA, NS, T <: Integer}

    # Precompute model shapes
    shape_s = source_shape(model)   # state shape tuple
    shape_a = action_shape(model)   # action shape tuple

    # Validate each action-state pair
    for (a, s) in sequence
        # check all entries >= 1
        @assert all(x -> x >= 1, s)
        @assert all(x -> x >= 1, a)

        # check each entry within bounds
        @assert all(((xi, yi),) -> xi <= yi, zip(s, shape_s))
        @assert all(((xi, yi),) -> xi <= yi, zip(a, shape_a))
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
