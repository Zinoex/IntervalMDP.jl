# Usage

The general procedure for using this package can be described in 5 steps
1. Construct a model, e.g. an Interval Markov Decision Process (IMDP) or some other subclass of [`FactoredRobustMarkovDecisionProcess`](@ref).
2. Choose property (reachability/reach-avoid/safety/reward + finite/infinite horizon).
3. Choose specification (optimistic/pessimistic + maximize/minimize + property).
4. Combine system and specification in a `VerificationProblem` or `ControlSynthesisProblem`, depending on whether you want to verify or synthesize a controller or not.
5. Call [`solve`](@ref) with the constructed problem and optionally a chosen algorithm. If no algorithm is given, a default algorithm will be chosen.

First, we construct a system; for the purpose of this example, we will construct either an IMDP. For more information about the different models, see [Models](@ref). Note that all subclasses of [`FactoredRobustMarkovDecisionProcess`](@ref) are converted to an fRMDP internally for verification and control synthesis, and the default algorithm is inferred based on the structure of the fRMDP.
An fRMDP consist of state variables (each can take on a finite number of values), action variables (similar to state variables), designated initial states, and a transition model; more specifically, the product of ambiguity sets for each marginal. See [Factored RMDPs](@ref) for more information about transition model.

An example of how to construct IMDP is the following:

```jldoctest usage
using IntervalMDP

# IMDP
prob1 = IntervalAmbiguitySets(;
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

prob2 = IntervalAmbiguitySets(;
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

prob3 = IntervalAmbiguitySets(;
    lower = [
        0.0 0.0
        0.0 0.0
        1.0 1.0
    ],
    upper = [
        0.0 0.0
        0.0 0.0
        1.0 1.0
    ],
)

transition_probs = [prob1, prob2, prob3]
initial_states = [1]  # Initial states are optional
mdp = IntervalMarkovDecisionProcess(transition_probs, initial_states)

# output

FactoredRobustMarkovDecisionProcess
├─ 1 state variables with cardinality: (3,)
├─ 1 action variables with cardinality: (2,)
├─ Initial states: [1]
├─ Transition marginals:
│  └─ Marginal 1:
│     ├─ Conditional variables: states = (1,), actions = (1,)
│     └─ Ambiguity set type: Interval (dense, Matrix{Float64})
└─Inferred properties
   ├─Model type: Interval MDP
   ├─Number of states: 3
   ├─Number of actions: 2
   ├─Default model checking algorithm: Robust Value Iteration
   └─Default Bellman operator algorithm: O-Maximization
```

Note that for an IMDP, the transition probabilities are specified as a list of transition probabilities (with each column representing an action) for each state.
The constructor will concatenate the transition probabilities into a single matrix, such that the columns represent source/action pairs and the rows represent target states.

!!! tip
    IMDPs can be very memory intensive if the ambiguity sets are stored as dense matrices. To reduce memory usage, consider using [Sparse matrices](@ref) and/or [Factored RMDPs](@ref) (recommended).

Next, we choose a property. Currently supported are reachability, reach-avoid, safety, reward, expected exit time and DFA-based properties.
For this example, we will use a reachability property, which requires specifying a set of target states `target_set`.
Furthermore, this package distinguishes distinguish between finite and infinite horizon properties - for finite horizon, a time horizon must be given, while for infinite horizon, a convergence threshold must be provided. 

In addition to the property, we need to specify whether we want to maximize or minimize the optimistic or pessimistic value (the value being satisfaction probability, discounted reward, etc.). We call this a specification.

```jldoctest usage
# Reachability
target_set = [3]
prop = FiniteTimeReachability(target_set, 10)  # Time steps
prop = InfiniteTimeReachability(target_set, 1e-6)  # Residual tolerance

## Specification
spec = Specification(prop, Pessimistic, Maximize)
spec = Specification(prop, Pessimistic, Minimize)
spec = Specification(prop, Optimistic, Maximize)
spec = Specification(prop, Optimistic, Minimize)

## Combine system and specification in a Problem
verification_problem = VerificationProblem(mdp, spec) # use `VerificationProblem(mdp, spec, strategy)` to verify under a given strategy
control_problem = ControlSynthesisProblem(mdp, spec)

# output

ControlSynthesisProblem
├─ FactoredRobustMarkovDecisionProcess
│  ├─ 1 state variables with cardinality: (3,)
│  ├─ 1 action variables with cardinality: (2,)
│  ├─ Initial states: [1]
│  ├─ Transition marginals:
│  │  └─ Marginal 1:
│  │     ├─ Conditional variables: states = (1,), actions = (1,)
│  │     └─ Ambiguity set type: Interval (dense, Matrix{Float64})
│  └─Inferred properties
│     ├─Model type: Interval MDP
│     ├─Number of states: 3
│     ├─Number of actions: 2
│     ├─Default model checking algorithm: Robust Value Iteration
│     └─Default Bellman operator algorithm: O-Maximization
└─ Specification
   ├─ Satisfaction mode: Optimistic
   ├─ Strategy mode: Minimize
   └─ Property: InfiniteTimeReachability
      ├─ Convergence threshold: 1.0e-6
      └─ Reach states: CartesianIndex{1}[CartesianIndex(3,)]
```

!!! tip
    For complex properties, e.g. LTLf, it is necessary to construct a Definite Finite Automaton (DFA) and (lazily) build the product with the fRMDP. See [Complex properties](@ref) for more details on the product construction and DFA properties. Note that constructing the DFA from an LTLf formula is currently not supported by this package. 

Finally, we call [`solve`](@ref) to solve the specification. `solve` returns the value function for all states in addition to the number of iterations performed and the last Bellman residual, wrapped in a solution object.

```jldoctest usage; output = false
sol = solve(verification_problem) # or solve(problem, alg) where e.g. alg = RobustValueIteration(LPMcCormickRelaxation()) to specify the algorithm
V, k, res = sol

# or alternatively
V, k, res = value_function(sol), num_iterations(sol), residual(sol)

# For control synthesis, we also get a strategy
sol = solve(control_problem)
V, k, res, strategy = sol

# output

IntervalMDP.ControlSynthesisSolution{StationaryStrategy{1, Vector{Tuple{Int32}}}, Float64, Vector{Float64}, Nothing}(StationaryStrategy{1, Vector{Tuple{Int32}}}(Tuple{Int32}[(2,), (2,), (1,)]), [0.9999977725460893, 0.9999985150307263, 1.0], [9.546231045653997e-7, 6.364154030435998e-7, -0.0], 37, nothing)
```
For now, only [`RobustValueIteration`](@ref) is supported, but more algorithms are planned.

!!! note
    To use multi-threading for parallelization, you need to either start julia with `julia --threads <n|auto>` where `n` is a positive integer or to set the environment variable `JULIA_NUM_THREADS` to the number of threads you want to use. For more information, see [Multi-threading](https://docs.julialang.org/en/v1/manual/multi-threading/).

## Sparse matrices
A disadvantage of IMDPs is that the size of the transition matrices grows ``O(n^2 m)`` where ``n`` is the number of states and ``m`` is the number of actions. Quickly, this becomes infeasible to store in memory. However, IMDPs frequently have lots of sparsity we may exploit. We choose in particular to store the transition matrices in the [Compressed Sparse Column (CSC)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)) format. This is a format that is widely used in Julia and other languages, and is supported by many linear algebra operations. The format consists of three arrays: `colptr`, `rowval` and `nzval`. The `colptr` array stores the indices of the first non-zero value in each column. The `rowval` array stores the row indices of the non-zero values, and the `nzval` array stores the non-zero values. We choose this format, since source states are stored as columns (see [`IntervalAmbiguitySets`](@ref) and [`Marginal`](@ref) for more information about the structure of the transition probability matrices). Thus the non-zero values for each source state is stored in sequentially in memory, enabling efficient memory access.

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
prob = IntervalAmbiguitySets(; lower = lower, upper = upper)
initial_state = 1
imc = IntervalMarkovChain(prob, initial_state)
```

If you know that the matrix can be built sequentially, you can use the `SparseMatrixCSC` constructor directly with `colptr`, `rowval` and `nzval`. This is more efficient, since `setindex!` of `SparseMatrixCSC` needs to perform a binary search to find the correct index to insert the value, and possibly expand the size of the array.

## CUDA
This package is supports GPU-accelerated value iteration via CUDA (only for [`IMDPs`](@ref) and [`IMCs`](@ref) at the moment). This includes not only trivial parallelization across states but also parallel algorithms for O-maximization within each state for better computational efficiency and coalesced memory access for more speed.

To use CUDA, you need to first install `CUDA.jl`. For more information about this, see [Installation](@ref).
Next, you need to load the package with the following command:
```julia
using CUDA
```

Loading CUDA will automatically load an extension that defines Bellman operators when the ambiguity sets are specified as CUDA arrays. It has been separated out into an extension to reduce precompilation time for users that do not need CUDA. Note that loading CUDA on a system without a CUDA-capable GPU, will not cause any errors, but only when running. You can check if CUDA is available using `CUDA.functional()`.

To use CUDA, you need to transfer the model to the GPU. Once on the GPU, you can use the same functions as the CPU implementation. Using Julia's multiple dispatch, the package will automatically dispatch to the appropriate implementation of the Bellman operators.

Similar to `CUDA.jl`, we provide a `cu` function that transfers the model to the GPU[^1]. You can either transfer the entire model or transfer the transition matrices separately. 
```julia
# Transfer entire model to GPU
prob = IntervalAmbiguitySets(;
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

# Transfer ambiguity sets to GPU
prob = IntervalMDP.cu(IntervalAmbiguitySets(;
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
))

mc = IntervalMarkovChain(prob, [1])

# Transfer transition matrices separately
prob = IntervalAmbiguitySets(;
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

mc = IntervalMarkovChain(prob, [1])
```

[^1]: The difference to `CUDA.jl`'s `cu` function is that `IntervalMDPs.jl`'s `cu` is opinionated to preserve value types and use `Int32` indices, to reduce register pressure but maintain accuracy