"""
    Sequential{VI <: Union{AllStates, AbstractVector}}

A type representing a sequential composition of Interval Markov Process.

Formally, let ``\\mathcal{M}_i = (S, S_0^i, \\bar{P}^i, \\underbar{P}^i)`` be an interval Markov process.
Then the sequential composition of ``\\mathcal{M}_1, \\ldots, \\mathcal{M}_n`` is defined as follows:
``(S, S_0, \\bar{P}, \\underbar{P})`` where ``S`` is the shared set of states, ``S_0`` is the set of initial states in ``S``,
``\\bar{P} = \\ldots `, and ``\\underbar{P} = \\ldots``.
# TODO: Update this definition when the theory is established.
This means that the transition probability from state ``s `` via ``s'`` to state ``s''`` is the product of the transition probabilities from ``s`` to ``s'`` in
the first process and from ``s'`` to ``s''`` in the second process.
Due to the sequential construction, value iteration can be done sequentially in reverse over the processes.

Then the `Sequential` type is defined by a vector of `IntervalMarkovProcess` and the initial states of the composition.

### Fields
- `sequential_processes::Vector{IntervalMarkovProcess}`: the list of processes in the composition.
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

"""
struct Sequential{
    VI <: InitialStates,
} <: SequentialIntervalMarkovProcess
    sequential_processes::Vector{IntervalMarkovProcess}
    initial_states::VI
    num_states::Int32
end

function Sequential(sequential_processes::Vector{IntervalMarkovProcess}, initial_states::InitialStates = AllStates())
    nstates = Int32(prod(num_states, sequential_processes))
    d = dims(first(sequential_processes))
    pns = product_num_states(first(sequential_processes)) |> recursiveflatten |> collect

    for process in sequential_processes
        if dims(process) != d
            throw(DimensionMismatch("Inconsistent number of dimensions."))
        end

        if collect(recursiveflatten(product_num_states(process))) != pns
            throw(DimensionMismatch("Inconsistent number of states per dimension."))
        end
    end
    
    return Sequential(
        sequential_processes,
        initial_states,
        nstates
    )
end

function Sequential(sequential_processes::Vector, initial_states::InitialStates = AllStates())
    sequential_processes = convert(Vector{IntervalMarkovProcess}, sequential_processes)
    
    return Sequential(
        sequential_processes,
        initial_states
    )
end

dims(mp::Sequential) = dims(first(sequential_processes(mp)))
product_num_states(mp::Sequential) = product_num_states(first(sequential_processes(mp)))

sequential_processes(mp::Sequential) = mp.sequential_processes
transition_matrix_type(mp::Sequential) = transition_matrix_type(first(sequential_processes(mp)))

# TODO: Implement check that if any of the processes are time-varying,
# then the time-varying processes have the same number of time steps.
