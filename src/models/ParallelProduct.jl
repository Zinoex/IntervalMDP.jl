"""
    ParallelProduct{VI <: Union{AllStates, AbstractVector}}

A type representing a parallel product composition of Interval Markov Process. See [1] for more information .

Formally, let ``\\mathcal{M}_i = (S^i, S_0^i, \\bar{P}^i, \\underbar{P}^i)`` be an interval Markov process.
Then the parallel product composition of ``\\mathcal{M}_1, \\ldots, \\mathcal{M}_n`` is defined as follows:
``(S, S_0, \\bar{P}, \\underbar{P})`` where ``S = S^1 \\times \\ldots \\times S^n``, ``S_0`` is the set of initial states in ``S`` (we ignore the initial states from the individual processes),
``\\bar{P} = \\bar{P}^1 \\cdot \\ldots \\cdot \\bar{P}^n``, and ``\\underbar{P} = \\underbar{P}^1 \\cdot \\ldots \\cdot \\underbar{P}^n``.
This means that the transition probability from state ``s = (s^1, \\ldots, s^n)`` to state ``s' = (s'^1, \\ldots, s'^n)`` is the product of the transition probabilities from ``s^i`` to ``s'^i`` in each process (accounting for actions if any of the processes are MDPs).
Due to the parallel construction, value iteration can be done independently for each process over the value function tensor.

Then the `ParallelProduct` type is defined by a vector of `IntervalMarkovProcess` (all have to be either stationary or time-varying) and the initial states of the composition.

### Fields
- `orthogonal_processes::Vector{IntervalMarkovProcess}`: the list of processes in the composition.
- `initial_states::VI`: initial states.
- `num_states::Int32`: number of states.

### Examples

# TODO: Update this example

```jldoctest
prob1 = IntervalProbabilities(;
    lower = [
        0.0 0.5 0.0
        0.1 0.3 0.0
        0.2 0.1 1.0
    ],
    upper = [
        0.5 0.7 0.0
        0.6 0.5 0.0
        0.7 0.3 1.0
    ],
)

prob2 = IntervalProbabilities(;
    lower = [
        0.2 0.1 0.0
        0.1 0.3 0.0
        0.0 0.5 1.0
    ],
    upper = [
        0.7 0.3 0.0
        0.6 0.5 0.0
        0.5 0.7 1.0
    ],
)

mc = TimeVaryingIntervalMarkovChain([prob1, prob2])
# or
initial_states = [1, 2, 3]
mc = TimeVaryingIntervalMarkovChain([prob1, prob2], initial_states)
```

[1] Nilsson, Petter, et al. "Toward Specification-Guided Active Mars Exploration for Cooperative Robot Teams." Robotics: Science and systems. Vol. 14. 2018.

"""
struct ParallelProduct{VI <: InitialStates} <: ProductIntervalMarkovProcess
    orthogonal_processes::Vector{IntervalMarkovProcess}
    initial_states::VI
    num_states::Int32
    subdims::Vector{Int32}
end

function ParallelProduct(
    orthogonal_processes::Vector{IntervalMarkovProcess},
    initial_states::InitialStates = AllStates(),
)
    nstates = Int32(prod(num_states, orthogonal_processes))
    sdims = map(dims, orthogonal_processes)

    return ParallelProduct(orthogonal_processes, initial_states, nstates, sdims)
end

function ParallelProduct(
    orthogonal_processes::Vector,
    initial_states::InitialStates = AllStates(),
)
    orthogonal_processes = convert(Vector{IntervalMarkovProcess}, orthogonal_processes)

    return ParallelProduct(orthogonal_processes, initial_states)
end

subdims(pp::ParallelProduct) = pp.subdims
dims(pp::ParallelProduct) = Int32(sum(subdims(pp)))

product_num_states(pp::ParallelProduct) = map(product_num_states, pp.orthogonal_processes)

first_transition_prob(pp::ParallelProduct) =
    first_transition_prob(first(pp.orthogonal_processes))
first_transition_prob(mp::SimpleIntervalMarkovProcess) = transition_prob(mp, 1)

max_actions(pp::ParallelProduct) = maximum(max_actions, orthogonal_processes(pp))

orthogonal_processes(pp::ParallelProduct) = pp.orthogonal_processes

# TODO: Implement check that if any of the processes are time-varying,
# then the time-varying processes have the same number of time steps.
