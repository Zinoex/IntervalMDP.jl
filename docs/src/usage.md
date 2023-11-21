# Usage

The general usage of this package can be described as 3 steps
1. Construct interval Markov process (IMC or IMDP)
2. Choose specification (reachability or reach-avoid)
3. Call `value_iteration` or `satisfaction_prob`.

First, we construct a system. We can either construct an interval Markov chain (IMC) or an interval Markov decision process. (IMDP)
Both systems consist of states, a designated initial state, and a transition matrix. In addition, an IMDP has actions. 
An example of how to construct either is the following:

```julia
# IMC
prob = IntervalProbabilities(;
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

initial_state = 1
mc = IntervalMarkovChain(prob, initial_state)

# IMDP
prob1 = IntervalProbabilities(;
    lower = [
        0.0 0.5
        0.1 0.3
        0.2 0.1
    ],
    upper = [
        0.5 0.7
        0.6 0.5
        0.7 0.3
    ],
)

prob2 = IntervalProbabilities(;
    lower = [
        0.1 0.2
        0.2 0.3
        0.3 0.4
    ],
    upper = [
        0.6 0.6
        0.5 0.5
        0.4 0.4
    ],
)

prob3 = IntervalProbabilities(;
    lower = [0.0; 0.0; 1.0],
    upper = [0.0; 0.0; 1.0]
)

transition_probs = [["a1", "a2"] => prob1, ["a1", "a2"] => prob2, ["sinking"] => prob3]
initial_state = 1
mdp = IntervalMarkovDecisionProcess(transition_probs, initial_state)
```

Note that for an IMDP, the transition probabilities are specified as a list of pairs from actions to transition probabilities for each state.
The constructor with concatenate the transition probabilities into a single matrix, such that the columns represent source/action pairs and the rows represent target states.
It will in addition construct a state pointer `stateptr` pointing to the first column of each state and concatenate a list of actions.
See [`IntervalMarkovDecisionProcess`](@ref) for more details on how to construct an IMDP.

For IMC, it is signifianctly simpler with source states on the columns and target states on the rows of the transition matrices.

Next, we choose a specification. Currently, we support reachability and reach-avoid specifications.
For reachability, we specify a target set of states and for reach-avoid we specify a target set of states and an avoid set of states.
Furthermore, we distinguish between finite and infinite horizon specifications.

```julia
# Reachability
target_set = [3]

spec = FiniteHorizonReachability(target_set, 10)  # Time steps
spec = InfiniteHorizonReachability(target_set, 1e-6)  # Residual tolerance

# Reach-avoid
target_set = [3]
avoid_set = [2]

spec = FiniteHorizonReachAvoid(target_set, avoid_set, 10)  # Time steps
spec = InfiniteHorizonReachAvoid(target_set, avoid_set, 1e-6)  # Residual tolerance

# Combine system and specification in a problem
problem = Problem(imdp_or_imc, spec)  # Default: Pessimistic
problem = Problem(imdp_or_imc, spec, Pessimistic)  # Explicit specification mode
```

Finally, we call `value_iteration` or `satisfaction_prob` to solve the specification.
`satisfaction_prob` returns the probability of satisfying the specification from the initial condition,
while `value_iteration` returns the value function for all states in addition to the number of iterations performed and the last Bellman residual.

```julia
V, k, residual = value_iteration(problem; upper_bound = false)
sat_prob = satisfaction_prob(problem)
```

!!! tip
    For less memory usage, it is recommended to use sparse matrix formats. Read [Sparse matrices](@ref) for more information.


## Sparse matrices
A disadvantage of IMDPs is that the size of the transition matrices grows ``O(n^2 m)`` where ``n`` is the number of states and ``m`` is the number of actions.
Quickly, this becomes infeasible to store in memory. However, IMDPs frequently have lots of sparsity we may exploit. We choose in particular to 
store the transition matrices in the [compressed sparse column (CSC)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)) format.
This is a format that is widely used in Julia and other languages, and is supported by many linear algebra operations.
It consists of three arrays: `colptr`, `rowval` and `nzval`. The `colptr` array stores the indices of the first non-zero value in each column.
The `rowval` array stores the row indices of the non-zero values, and the `nzval` array stores the non-zero values.
We choose this format, since source states are on the columns (see [`IntervalProbabilities`](@ref) for more information about the structure of the transition probability matrices).
Thus the non-zero values for each source state is stored in sequentially in memory, enabling efficient memory access.

To use `SparseMatrixCSC`, we need to load `SparseArrays`. Below is an example of how to construct an `IntervalMarkovChain` with sparse transition matrices.
```@example
using SparseArrays

lower = spzeros(3, 3)
lower[2, 1] = 0.1
lower[3, 1] = 0.2
lower[1, 2] = 0.5
lower[2, 2] = 0.3
lower[3, 2] = 0.1
lower[3, 3] = 1.0

lower
```

```@setup sparse
using SparseArrays
```

```@example sparse
upper = spzeros(3, 3)
upper[1, 1] = 0.5
upper[2, 1] = 0.6
upper[3, 1] = 0.7
upper[1, 2] = 0.7
upper[2, 2] = 0.5
upper[3, 2] = 0.3
upper[3, 3] = 1.0

upper
```

```julia
prob = IntervalProbabilities(; lower = lower, upper = upper)
initial_state = 1
mc = IntervalMarkovChain(prob, initial_state)

```

If you know that the matrix can be built sequentially, you can use the `SparseMatrixCSC` constructor directly with `colptr`, `rowval` and `nzval`.
This is more efficient, since `setindex!` of `SparseMatrixCSC` needs to perform a binary search to find the correct index to insert the value,
and possibly expand the size of the array.

## CUDA
- Pacakges that are required for CUDA to work
- Transfer matrices vs transfer model