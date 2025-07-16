# Usage

The general procedure for using this package can be described in 3 steps
1. Construct interval Markov process (IMC or IMDP)
2. Choose property (reachability, reach-avoid, safety, or reward + finite/infinite horizon)
3. Choose specification (optimistic/pessimistic, maximize/minimize + property)
3. Call `value_iteration` or `control_synthesis`.

First, we construct a system. We can either construct an interval Markov chain (IMC) or an interval Markov decision process. (IMDP)
Both systems consist of states, a designated initial state, and a transition matrix. In addition, an IMDP has actions. 
An example of how to construct either is the following:

```julia
using IntervalMDP

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

initial_states = [1]  # Initial states are optional
mc = IntervalMarkovChain(prob, initial_states)

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

transition_probs = [prob1, prob2, prob3]
initial_states = [1]  # Initial states are optional
imdp = IntervalMarkovDecisionProcess(transition_probs, initial_states)
```

Note that for an IMDP, the transition probabilities are specified as a list of transition probabilities (with each column representing an action) for each state.
The constructor will concatenate the transition probabilities into a single matrix, such that the columns represent source/action pairs and the rows represent target states.
It will in addition construct a state pointer `stateptr` pointing to the first column of each state.
See [`IntervalMarkovDecisionProcess`](@ref) for more details on how to construct an IMDP.

For IMC, the transition probability structure is significantly simpler with source states on the columns and target states on the rows of the transition matrices. Internally, they are both represented by an `IntervalMarkovDecisionProcess`.

Next, we choose a property. Currently supported are reachability, reach-avoid, safety, and reward properties.
For reachability, we specify a target set of states and for reach-avoid we specify a target set of states and an avoid set of states.
For a safety property, we specify a set of states that must be avoided, and for a reward property, we specify a reward matrix and a discount factor.
Furthermore, this package distinguishes distinguish between finite and infinite horizon properties - for finite horizon, a time horizon must be given while for infinite horizon, a convergence threshold must be given. In addition to the property, we need to specify whether we want to maximize or minimize the optimistic or pessimistic satisfaction probability or discounted reward.

```julia
## Properties
# Reachability
target_set = [3]

prop = FiniteTimeReachability(target_set, 10)  # Time steps
prop = InfiniteTimeReachability(target_set, 1e-6)  # Residual tolerance

# Reach-avoid
target_set = [3]
avoid_set = [2]

prop = FiniteTimeReachAvoid(target_set, avoid_set, 10)  # Time steps
prop = InfiniteTimeReachAvoid(target_set, avoid_set, 1e-6)  # Residual tolerance

# Safety
avoid_set = [2]

prop = FiniteTimeSafety(avoid_set, 10)  # Time steps
prop = InfiniteTimeSafety(avoid_set, 1e-6)  # Residual tolerance

# Reward
reward = [1.0, 2.0, 3.0]
discount = 0.9  # Has to be between 0 and 1

prop = FiniteTimeReward(reward, discount, 10)  # Time steps
prop = InfiniteTimeReward(reward, discount, 1e-6)  # Residual tolerance

## Specification
spec = Specification(prop, Pessimistic, Maximize)
spec = Specification(prop, Pessimistic, Minimize)
spec = Specification(prop, Optimistic, Maximize)
spec = Specification(prop, Optimistic, Minimize)

## Combine system and specification in a Problem
problem = VerificationProblem(imdp_or_imc, spec)
```

Finally, we call [`solve`](@ref) to solve the specification. `solve` returns the value function for all states in addition to the number of iterations performed and the last Bellman residual, wrapped in a solution object.

```julia
sol = solve(problem) # or solve(problem, RobustValueIteration())
V, k, res = sol

# or alternatively
V, k, res = value_function(sol), num_iterations(sol), residual(sol)
```
For now, only [`RobustValueIteration`](@ref) is supported, but more algorithms are planned.

!!! note
    To use multi-threading for parallelization, you need to either start julia with `julia --threads <n|auto>` where `n` is a positive integer or to set the environment variable `JULIA_NUM_THREADS` to the number of threads you want to use. For more information, see [Multi-threading](https://docs.julialang.org/en/v1/manual/multi-threading/).

!!! tip
    For less memory usage, it is recommended to use [Sparse matrices](@ref) and/or [Orthogonal models](@ref).

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
imc = IntervalMarkovChain(prob, initial_state)
```

If you know that the matrix can be built sequentially, you can use the `SparseMatrixCSC` constructor directly with `colptr`, `rowval` and `nzval`.
This is more efficient, since `setindex!` of `SparseMatrixCSC` needs to perform a binary search to find the correct index to insert the value,
and possibly expand the size of the array.

## Orthogonal models
TODO

## Control synthesis
TODO

## CUDA
Part of the innovation of this package is GPU-accelerated value iteration via CUDA. This includes not only
trivial parallelization across states but also parallel algorithms for O-maximization within each state
for better computational efficiency and coalesced memory access for more speed. 

To use CUDA, you need to first install `CUDA.jl`. For more information about this, see [Installation](@ref).
Next, you need to load the package with the following command:
```julia
using CUDA
```

Loading CUDA will automatically load an extension that defines value iteration with CUDA arrays.
It has been separated out into an extension to reduce precompilation time for users that do not need CUDA.
Note that loading CUDA on a system without a CUDA-capable GPU, will not cause any errors, although a warning, upon loading, but only when running.
You can check if CUDA is correctly loaded using `CUDA.functional()`.

To use CUDA, you need to transfer the model to the GPU. Once on the GPU, you can use the same functions as the CPU implementation.
Using Julia's multiple dispatch, the package will automatically dispatch to the appropriate implementation of `bellman!`.

Similar to `CUDA.jl`, we provide a `cu` function that transfers the model to the GPU[^1]. You can either transfer the entire model
or transfer the transition matrices separately. 
```julia
# Transfer entire model to GPU
prob = IntervalProbabilities(;
    lower = sparse_hcat(
        SparseVector(3, [2, 3], [0.1, 0.2]),
        SparseVector(3, [1, 2, 3], [0.5, 0.3, 0.1]),
        SparseVector(3, [3], [1.0]),
    ),
    upper = sparse_hcat(
        SparseVector(3, [1, 2, 3], [0.5, 0.6, 0.7]),
        SparseVector(3, [1, 2, 3], [0.7, 0.5, 0.3]),
        SparseVector(3, [3], [1.0]),
    ),
)

mc = IntervalMDP.cu(IntervalMarkovChain(prob, 1))

# Transfer transition matrices separately
prob = IntervalProbabilities(;
    lower = IntervalMDP.cu(sparse_hcat(
        SparseVector(3, [2, 3], [0.1, 0.2]),
        SparseVector(3, [1, 2, 3], [0.5, 0.3, 0.1]),
        SparseVector(3, [3], [1.0]),
    )),
    upper = IntervalMDP.cu(sparse_hcat(
        SparseVector(3, [1, 2, 3], [0.5, 0.6, 0.7]),
        SparseVector(3, [1, 2, 3], [0.7, 0.5, 0.3]),
        SparseVector(3, [3], [1.0]),
    )),
)

mc = IntervalMarkovChain(prob,[1])
```

[^1]: The difference to `CUDA.jl`'s `cu` function is that `IntervalMDPs.jl`'s `cu` is opinoinated to `Float64` values and `Int32` indices, to reduce register pressure but maintain accuracy