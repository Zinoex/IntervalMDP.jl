struct ProductIterator{AI, SI}
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

struct ZipIterator{AI, SI}
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
Base.getindex(iter::ZipIterator, i) = begin
    A = iter.A
    S = iter.S

    return (return (A[i], S[i]))
end

Base.iterate(iter::ZipIterator) = begin
    iter.n == 0 && return nothing

    A = iter.A
    S = iter.S

    i = firstindex(A)
    return ((A[i], S[i]), i)
end

Base.iterate(iter::ZipIterator, i) = begin
    A = iter.A
    S = iter.S

    i += 1
    if i > iter.n
        return nothing
    end

    return ((A[i], S[i]), i)
end

struct OnPolicyActionIterator
    S::CartesianIndices
    strategy_cache::AbstractStrategyCache

    function OnPolicyActionIterator(S::CartesianIndices, strategy_cache::AbstractStrategyCache)
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



abstract type SamplingStrategy end
abstract type ThreadedSamplingStrategy <: SamplingStrategy end

function sample(::SamplingStrategy, model) end

struct AllSampling <: SamplingStrategy end
struct ThreadedAllSampling <: ThreadedSamplingStrategy end

default_sampling_strategy() = AllSampling()
default_sampling_strategy(::NotThreaded) = AllSampling()
default_sampling_strategy(::IsThreaded) = ThreadedAllSampling()

sample(::AllSampling, model) = exhaustive_cartesian(model, NotThreaded())
sample(::ThreadedAllSampling, model) = exhaustive_cartesian(model, IsThreaded())

sample(::AllSampling, model, strategy_cache::AbstractStrategyCache) = exhaustive_cartesian(model, strategy_cache, NotThreaded())
sample(::ThreadedAllSampling, model, strategy_cache::AbstractStrategyCache) = exhaustive_cartesian(model, strategy_cache, IsThreaded())


exhaustive_cartesian(model::FactoredRMDP, threaded::ThreadedType) = exhaustive_cartesian(model, modeltype(model), threaded)
exhaustive_cartesian(model::FactoredRMDP, ::IsIMDP, ::NotThreaded) = ProductIterator(CartesianIndices(action_shape(model)), CartesianIndices(source_shape(model)))
exhaustive_cartesian(model::IntervalAmbiguitySets, ::NotThreaded) = ProductIterator(CartesianIndices(action_shape(model)), CartesianIndices(source_shape(model)))
function exhaustive_cartesian(model::FactoredRMDP, ::IsIMDP, ::IsThreaded) 
    A = CartesianIndices(action_shape(model))
    S = CartesianIndices(source_shape(model))

    return ProductIterator(A, S)
end
function exhaustive_cartesian(model::IntervalAmbiguitySets, ::IsThreaded) 
    A = CartesianIndices(action_shape(model))
    S = CartesianIndices(source_shape(model))

    return ProductIterator(A, S)
end

exhaustive_cartesian(model, strategy_cache::OptimizingStrategyCache, threaded::ThreadedType) = exhaustive_cartesian(model, threaded)
exhaustive_cartesian(model::FactoredRMDP, strategy_cache::NonOptimizingStrategyCache, threaded::ThreadedType) = exhaustive_cartesian(model, modeltype(model), strategy_cache, threaded)

function exhaustive_cartesian(model::FactoredRMDP, ::IsIMDP, strategy_cache::NonOptimizingStrategyCache, ::NotThreaded)

    S = CartesianIndices(source_shape(model))
    A = OnPolicyActionIterator(S, strategy_cache)

    return ZipIterator(A, S)
end

function exhaustive_cartesian(model::FactoredRMDP, ::IsIMDP, strategy_cache::NonOptimizingStrategyCache, ::IsThreaded)

    S = CartesianIndices(source_shape(model))
    A = OnPolicyActionIterator(S, strategy_cache)

    return ZipIterator(A, S)
end

function exhaustive_cartesian(model::IntervalAmbiguitySets, strategy_cache::NonOptimizingStrategyCache, ::NotThreaded) 
    S = CartesianIndices(source_shape(model))
    A = OnPolicyActionIterator(S, strategy_cache)

    return ZipIterator(A, S)
end

function exhaustive_cartesian(model::IntervalAmbiguitySets, strategy_cache::NonOptimizingStrategyCache, ::IsThreaded)
    S = CartesianIndices(source_shape(model))
    A = OnPolicyActionIterator(S, strategy_cache)

    return ZipIterator(A, S) 
end



struct GivenSequence{} <: SamplingStrategy
    sequence::Vector{Tuple{Int}}
end

function sample(strategy::GivenSequence, 
                model, 
                sequence::Vector{Tuple{NTuple{N, T}, NTuple{M, T}}} # each element: (state_tuple, action_tuple
                ) where {N, M, T<:Integer}
    return custom_sequence(model, sequence)
end

function custom_sequence(
    model::FactoredRMDP,
    sequence::Vector{Tuple{NTuple{N, T}, NTuple{M, T}}}
)::AbstractVector{Tuple{CartesianIndex{N}, CartesianIndex{M}}} where {N, M, T<:Integer}

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

    # Convert to CartesianIndex tuples
    return [(CartesianIndex(a...), CartesianIndex(s...)) for (s, a) in sequence]
end

# TODO: 1. random sampling of states, with or without replacement, with or without weighting (e.g. based on current value function)
# TODO:     - subset of states each iteration?
# TODO:     - one state per iteration?
# TODO:
# TODO: 2. (epsilon) greedy on policy trajectory simulation
# TODO: 3. BRTDP gap based trajectory simulation
# TODO: 


### Robust Value Iteration
sampling_strategy(alg::RobustValueIteration, ::NotThreaded) = AllSampling()
sampling_strategy(alg::RobustValueIteration, ::IsThreaded) = ThreadedAllSampling()
