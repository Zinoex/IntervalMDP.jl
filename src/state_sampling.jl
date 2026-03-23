struct ThreadedProductIterator{AI, SI}
    A::AI
    S::SI
    nA::Int
    nS::Int

    function ThreadedProductIterator(A, S)
        nA = length(A)
        nS = length(S)
        new{typeof(A), typeof(S)}(A, S, nA, nS)
    end
end

Base.length(iter::ThreadedProductIterator) = iter.nA * iter.nS
Base.firstindex(iter::ThreadedProductIterator) = 1
Base.getindex(iter::ThreadedProductIterator, i) = begin
    A = iter.A
    S = iter.S

    nA = iter.nA
    nS = iter.nS

    ia = ((i - 1) % nA) + firstindex(A)
    is = ((i - 1) ÷ nA) + firstindex(S)
    return (A[ia], S[is])
end

struct ThreadedIndexIterator{AI, SI}
    A::AI
    S::SI
    n::Int

    function ThreadedIndexIterator(A, S)
        nA = length(A)
        nS = length(S)

        @assert nA == nS "Action and state spaces must have the same length for ThreadedIndexIterator"

        new{typeof(A), typeof(S)}(A, S, nA)
    end
end

Base.length(iter::ThreadedIndexIterator) = iter.n
Base.firstindex(iter::ThreadedIndexIterator) = 1
Base.getindex(iter::ThreadedIndexIterator, i) = begin
    A = iter.A
    S = iter.S

    return (return (A[i], S[i]))
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
exhaustive_cartesian(model::FactoredRMDP, ::IsIMDP, ::NotThreaded) = Iterators.product(CartesianIndices(action_shape(model)), CartesianIndices(source_shape(model)))
exhaustive_cartesian(model::IntervalAmbiguitySets, ::NotThreaded) = Iterators.product(CartesianIndices(action_shape(model)), CartesianIndices(source_shape(model)))
function exhaustive_cartesian(model::FactoredRMDP, ::IsIMDP, ::IsThreaded) 
    A = CartesianIndices(action_shape(model))
    S = CartesianIndices(source_shape(model))

    return ThreadedProductIterator(A, S)
end
function exhaustive_cartesian(model::IntervalAmbiguitySets, ::IsThreaded) 
    A = CartesianIndices(action_shape(model))
    S = CartesianIndices(source_shape(model))

    return ThreadedProductIterator(A, S)
end

exhaustive_cartesian(model, strategy_cache::OptimizingStrategyCache, threaded::ThreadedType) = exhaustive_cartesian(model, threaded)
exhaustive_cartesian(model::FactoredRMDP, strategy_cache::NonOptimizingStrategyCache, threaded::ThreadedType) = exhaustive_cartesian(model, modeltype(model), strategy_cache, threaded)

function exhaustive_cartesian(model::FactoredRMDP, ::IsIMDP, strategy_cache::NonOptimizingStrategyCache, ::NotThreaded)

    S = CartesianIndices(source_shape(model))

    return (
        (CartesianIndex(strategy_cache[jₛ]), jₛ)
        for jₛ in S
    )
    # map(jₛ -> (CartesianIndex(strategy_cache[jₛ]), jₛ), CartesianIndices(source_shape(model)))
end

function exhaustive_cartesian(model::FactoredRMDP, ::IsIMDP, strategy_cache::NonOptimizingStrategyCache, ::IsThreaded)

    S = CartesianIndices(source_shape(model))
    A = (CartesianIndex(strategy_cache[jₛ]) for jₛ in S)
    
    return ThreadedIndexIterator(A, S)
end

function exhaustive_cartesian(model::IntervalAmbiguitySets, strategy_cache::NonOptimizingStrategyCache, ::NotThreaded) 
    S = CartesianIndices(source_shape(model))

    return (
        (CartesianIndex(strategy_cache[jₛ]), jₛ)
        for jₛ in S
    )
    # map(jₛ -> (CartesianIndex(strategy_cache[jₛ]), jₛ), CartesianIndices(source_shape(model)))
end

function exhaustive_cartesian(model::IntervalAmbiguitySets, strategy_cache::NonOptimizingStrategyCache, ::IsThreaded)
    S = CartesianIndices(source_shape(model))
    A = (CartesianIndex(strategy_cache[jₛ]) for jₛ in S)
    
    return ThreadedIndexIterator(A, S) 
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
