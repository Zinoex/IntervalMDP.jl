# Models 
#### Mathematical Notation 
We denote the natural numbers by ``\mathbb{N}`` and ``\mathbb{N}_0 = \mathbb{N} \cup \{0\}``. A probability distribution ``\gamma`` over a finite set ``S`` is a function ``\gamma : S \to [0, 1]`` satisfying ``\sum_{s \in S} \gamma(s) = 1``. The support of the distribution ``\mathop{supp}(\gamma)`` is defined as ``\mathop{supp}(\gamma) = \{ s \in S : \gamma(s) > 0\}``. We denote by ``\mathcal{D}(S)`` the set of all probability distributions over ``S``.
For ``\underline{\gamma}, \overline{\gamma} : S \to [0, 1]`` such that ``\underline{\gamma}(s) \leq \overline{\gamma}(s)`` for each ``s \in S`` and ``\sum_{s \in S} \underline{\gamma}(s) \leq 1 \leq \sum_{s \in S} \overline{\gamma}(s)``, an interval ambiguity set ``\Gamma \subset \mathcal{D}(S)`` is the set of distributions such that 
```math
    \Gamma = \{ \gamma \in \mathcal{D}(S) \,:\, \underline{\gamma}(s) \leq \gamma(s) \leq \overline{\gamma}(s) \text{ for each } s \in S \}.
```
``\underline{\gamma}, \overline{\gamma}`` are referred to as the interval bounds of the interval ambiguity set.
For ``n`` finite sets ``S_1, \ldots, S_n`` we denote by ``S_1 \times \cdots \times S_n`` their Cartesian product. Given ``S = S_1 \times \cdots \times S_n`` and ``n`` ambiguity sets ``\Gamma_i \in \mathcal{D}(S_i)``, ``i = 1, \ldots, n``, the product ambiguity set ``\Gamma \subseteq \mathcal{D}(S)`` is defined as: 
```math
    \Gamma = \left\{ \gamma \in \mathcal{D}(S) \,:\, \gamma(s) = \prod_{i=1}^n \gamma^i(s^i), \, \gamma^i \in \Gamma_i \right\}
```
where ``s = (s_1, \ldots, s_n) \in S``. We will denote the product ambiguity set as ``\Gamma = \bigotimes_{i=1}^n \Gamma_i``. Each ``\Gamma_i`` is called a marginal or component ambiguity set. A transition is a triplet ``(s, a, t) \in S \times A \times S`` where ``s`` is the source state, ``a`` is the action, and ``t`` is the target state.

## Factored RMDPs
Factored Robust Markov Decision Processes (fRMDPs) [schnitzer2025efficient, delgado2011efficient](@cite) are an extension of Robust Markov Decision Processes (RMDPs) [nilim2005robust, wiesemann2013robust, suilen2024robust](@cite) that incorporate a factored representation of the state and action spaces, i.e. with state and action variables. This allows for a more compact representation of the transition model and flexibility in modeling complex systems. First, we define here fRMDPs, and then in the subsequent sections, we define various special subclasses of fRMDPs, including how they relate to each other and to fRMDPs.

Formally, a fRMDP ``M`` is a tuple ``M = (S, S_0, A, \mathcal{G}, \Gamma)``, where

- ``S = S_1 \times \cdots \times S_n`` is a finite set of joint states with ``S_i`` being a finite set of states for the ``i``-th state variable,
- ``S_0 \subseteq S`` is a set of initial states,
- ``A = A_1 \times \cdots \times A_m`` is a finite set of joint actions with ``A_j`` being a finite set of actions for the ``j``-th action variable,
- ``\mathcal{G} = (\mathcal{V}, \mathcal{E})`` is a directed bipartite graph with nodes ``\mathcal{V} = \mathcal{V}_{ind} \cup \mathcal{V}_{cond} = \{S_1, \ldots, S_n, A_1, \ldots, A_m\} \cup \{S'_1, \ldots, S'_n\}`` representing the state and action variables and their next-state counterparts, and edges ``\mathcal{E} \subseteq \mathcal{V}_{ind} \times \mathcal{V}_{cond}`` representing dependencies of ``S'_i`` on ``S_j`` and ``A_k``,
- ``\Gamma = \{\Gamma_{s, a}\}_{s \in S, a \in A}`` is a set of ambiguity sets for source-action pair ``(s, a)``, where each ``\Gamma_{s, a} = \bigotimes_{i=1}^n \Gamma^i_{\text{Pa}_\mathcal{G}(S'_i) \cap (s, a)}`` is a product of ambiguity sets ``\Gamma^i_{\text{Pa}_\mathcal{G}(S'_i) \cap (s, a)}`` along each marginal ``i`` conditional on the values in ``(s, a)`` of the parent variables ``\text{Pa}_\mathcal{G}(S'_i)`` of ``S'_i`` in ``\mathcal{G}``, i.e.
```math
    \Gamma_{s, a} = \left\{ \gamma \in \mathcal{D}(S) \,:\, \gamma(t) = \prod_{i=1}^n \gamma^i(t_i | s_{\text{Pa}_{\mathcal{G}_S}(S'_i)}, a_{\text{Pa}_{\mathcal{G}_A}(S'_i)}), \, \gamma^i(\cdot | s_{\text{Pa}_{\mathcal{G}_S}(S'_i)}, a_{\text{Pa}_{\mathcal{G}_A}(S'_i)}) \in \Gamma^i_{\text{Pa}_\mathcal{G}(S'_i)} \right\}.
```

For a given source-action pair ``(s, a) \in S \times A``, any distribution ``\gamma_{s, a} \in \Gamma_{s, a}`` is called a feasible distribution, and feasible transitions are triplets ``(s, a, t) \in S \times A \times S`` where ``t \in \mathop{supp}(\gamma_{s, a})`` for any feasible distribution ``\gamma_{s, a} \in \Gamma_{s, a}``. A path of an fRMDP is a sequence of states and actions ``\omega = s[0], a[0], s[1], a[1], \dots`` where ``s[k] \in S`` and ``a[k] \in A`` for all ``k \in \mathbb{N}_0``, and ``(s[k], a[k], s[k + 1])`` is a feasible transition  for all ``k \in \mathbb{N}_0``. We denote by ``\omega[k] = s[k]`` the state of the path at time ``k \in \mathbb{N}_0`` and by ``\Omega`` and ``\Omega_{fin}`` the set of all infinite and finite paths, respectively.

A _strategy_ or _policy_ for an fRMDP is a function ``\pi : \Omega_{fin} \to A`` that assigns an action, given a (finite) path called the history. _Time-dependent_ Markov strategies are functions from state and time step to an action, i.e. ``\pi : S \times \mathbb{N}_0 \to A``. This can equivalently be described as a sequence of functions indexed by time ``\mathbf{\pi} = (\pi[0], \pi[1], \ldots)``. If ``\pi`` does not depend on time and solely depends on the current state, it is called a _stationary_ strategy. Similar to a strategy, an adversary ``\eta`` is a function that assigns a feasible distribution to a given state. The focus of this package is on dynamic uncertainties where the choice of the adversary is resolved at every time step, called dynamic uncertainty, and where the adversary has access to both the current state and action, called ``(s, a)``-rectangularity. We refer to [suilen2024robust](@cite) for further details on the distinction between static and dynamic uncertainties, types of rectangularity, and their implications. Given a strategy and an adversary, an fRMDP collapses to a finite (factored) Markov chain.

Below is an example of how to construct an fRMDP with 2 state variables (2 and 3 values respectively) and 2 action variables (1 and 2 values respectively), where each marginal ambiguity set is an interval ambiguity set. The first marginal depends on both state variables and the first action variable, while the second marginal only depends on the second state variable and the second action variable.
```@example
using IntervalMDP # hide

state_vars = (2, 3)
action_vars = (1, 2)

state_indices = (1, 2)
action_indices = (1,)
state_dims = (2, 3)
action_dims = (1,)
marginal1 = Marginal(IntervalAmbiguitySets(;
    # 6 ambiguity sets = 2 * 3 source states, 1 action
    # Column layout: (a¹₁, s¹₁, s²₁), (a¹₁, s¹₂, s²₁), (a¹₁, s¹₁, s²₂), (a¹₁, s¹₂, s²₂), (a¹₁, s¹₁, s²₃), (a¹₁, s¹₂, s²₃)
    # Equivalent to CartesianIndices(actions_dims..., state_dims...), i.e. actions first, then states in lexicographic order
    lower = [
        1/15  7/30  1/15  13/30  4/15  1/6
        2/5   7/30  1/30  11/30  2/15  1/10
    ],
    upper = [
        17/30  7/10   2/3   4/5  7/10  2/3
        9/10   13/15  9/10  5/6  4/5   14/15
    ]
), state_indices, action_indices, state_dims, action_dims)

state_indices = (2,)
action_indices = (2,)
state_dims = (3,)
action_dims = (2,)
marginal2 = Marginal(IntervalAmbiguitySets(;
    # 6 ambiguity sets = 3 source states, 2 actions
    # Column layout: (a²₁, s²₁), (a²₂, s²₁), (a²₁, s²₂), (a²₂, s²₂), (a²₁, s²₃), (a²₂, s²₃)
    # Equivalent to CartesianIndices(actions_dims..., state_dims...), i.e. actions first, then states in lexicographic order
    lower = [
        1/30  1/3   1/6   1/15  2/5   2/15
        4/15  1/4   1/6   1/30  2/15  1/30
        2/15  7/30  1/10  7/30  7/15  1/5
    ],
    upper = [
        2/3    7/15  4/5    11/30  19/30  1/2
        23/30  4/5   23/30  3/5    7/10   8/15
        7/15   4/5   23/30  7/10   7/15   23/30
    ]
), state_indices, action_indices, state_dims, action_dims)

initial_states = [(1, 1)]  # Initial states are optional
mdp = FactoredRobustMarkovDecisionProcess(state_vars, action_vars, (marginal1, marginal2), initial_states)
```

!!! warn
    Notice that source-action pairs are on the columns of the matrices to defined the interval bounds. This is counter to most literature on transition matrices where transitions are from row to column. The choice of layout is to ensure that the memory access pattern is cache-friendly, as each column is stored contiguously in memory (column-major) and the Bellman updates iterate outer-most over source-action pairs. However, it also has a fundamental mathematical justification: the transition matrix can be viewed as a linear operator and the matrix form of a linear operator is defined such that the columns correspond to the input dimensions, i.e. from column to row. Furthermore, actions for the same source state are stored contiguously, which is also important for cache efficiency.

## IMCs
Interval Markov Chains (IMCs) [delahaye2011decision](@cite) are a subclass of fRMDPs and a generalization of Markov Chains (MCs), where the transition probabilities are not known exactly, but they are constrained to be in some probability interval.
Formally, an IMC ``M`` is a tuple ``M = (S, S_0, \Gamma)``, where

- ``S`` is a finite set of states,
- ``S_0 \subseteq S`` is a set of initial states,
- ``\Gamma = \{\Gamma_{s}\}_{s \in S}`` is a set of ambiguity sets for source state ``s``, where each ``\Gamma_{s}`` is an interval ambiguity set over ``S``.

An IMC is equivalent to an fRMDP where there is only one state variable, no action variables, and the ambiguity sets are interval ambiguity sets. The dependency graph is just two nodes ``S`` and ``S'`` with a single edge from the former to the latter. Paths and adversaries are defined similarly to fRMDPs.

Example:
```@example
using IntervalMDP # hide

prob = IntervalAmbiguitySets(;
    lower = [
        0     1/2   0
        1/10  3/10  0
        1/5   1/10  1
    ],
    upper = [
        1/2   7/10  0
        3/5   1/2   0
        7/10  3/10  1
    ],
)

initial_states = [1]  # Initial states are optional
mc = IntervalMarkovChain(prob, initial_states)
```

## IMDPs
Interval Markov Decision Processes (IMDPs) [givan2000bounded, lahijanian2015formal](@cite), also called bounded-parameter MDPs, are a subclass of fRMDPs and a generalization of MDPs, where the transition probabilities, given source state and action, are not known exactly, but they are constrained to be in some probability interval. IMDPs generalized IMCs by adding actions.
Formally, an IMDP ``M`` is a tuple ``M = (S, S_0, A, \Gamma)``, where

- ``S`` is a finite set of states,
- ``S_0 \subseteq S`` is a set of initial states,
- ``A`` is a finite set of actions,
- ```\Gamma = \{\Gamma_{s, a}\}_{s \in S, a \in A}`` is a set of ambiguity sets for source-action pair ``(s, a)``, where each ``\Gamma_{s, a}`` is an interval ambiguity set over ``S``.

An IMDP is equivalent to an fRMDP where there is only one state variable, one action variable, and the ambiguity sets are interval ambiguity sets. The dependency graph is three nodes ``S``, ``A``, and ``S'`` with two edges ``S \rightarrow S'`` and ``A \rightarrow S'``. Paths and adversaries are defined similarly to fRMDPs.

Example:
```@example
using IntervalMDP # hide

prob1 = IntervalAmbiguitySets(;
    lower = [
        0    1/2
        1/10 3/10
        1/5  1/10
    ],
    upper = [
        1/2  7/10
        3/5  1/2
        7/10 3/10
    ],
)

prob2 = IntervalAmbiguitySets(;
    lower = [
        1/10 1/5
        1/5  3/10
        3/10 2/5
    ],
    upper = [
        3/5 3/5
        1/2 1/2
        2/5 2/5
    ],
)

prob3 = IntervalAmbiguitySets(;
    lower = Float64[
        0 0
        0 0
        1 1
    ],
    upper = Float64[
        0 0
        0 0
        1 1
    ]
)

initial_states = [1]
mdp = IntervalMarkovDecisionProcess([prob1, prob2, prob3], initial_states)

# alternatively
prob = IntervalAmbiguitySets(;
    lower = [
        0    1/2  1/10 1/5  0 0
        1/10 3/10 1/5  3/10 0 0
        1/5  1/10 3/10 2/5  1 1
    ],
    upper = [
        1/2  7/10 3/5 2/5 0 0
        3/5  1/2  1/2 2/5 0 0
        7/10 3/10 2/5 2/5 1 1
    ],
)

num_actions = 2
mdp = IntervalMarkovDecisionProcess(prob, num_actions, initial_states)
```

It is possible to skip defining actions when the transition is a guaranteed self-loop and is the last states in the ambiguity set. 
This is useful for defining target states in reachability problems. The example below has 3 states (as shown by the 3 rows) and 2 actions
(explictly defined by `num_actions = 2`). The last state is a target state with a guaranteed self-loop, i.e., the transition probabilities are ``P(3 | 3, a) = 1`` for both actions ``a \in \{1, 2\}``.
```@example
using IntervalMDP # hide

prob = IntervalAmbiguitySets(;
    lower = [
        0    1/2  1/10 1/5
        1/10 3/10 1/5  3/10
        1/5  1/10 3/10 2/5 
    ],
    upper = [
        1/2  7/10 3/5 2/5
        3/5  1/2  1/2 2/5
        7/10 3/10 2/5 2/5
    ],
)

num_actions = 2
mdp = IntervalMarkovDecisionProcess(prob, num_actions)
```

## odIMDPs
Orthogonally-decoupled IMDPs (odIMDPs) [mathiesen2025scalable](@cite) are a subclass of fRMDPs designed to be more memory-efficient than IMDPs. The states are structured into an orthogonal, or grid-based, decomposition and the transition probability ambiguity sets, for each source-action pair, as a product of interval ambiguity sets along each marginal. 

Formally, an odIMDP ``M`` with ``n`` marginals is a tuple ``M = (S, S_0, A, \Gamma)``, where

- ``S = S_1 \times \cdots \times S_n`` is a finite set of joint states with ``S_i`` being a finite set of states for the ``i``-th marginal,
- ``S_0 \subseteq S`` is a set of initial states,
- ``A`` is a finite set of actions,
- ``\Gamma = \{\Gamma_{s, a}\}_{s \in S, a \in A}`` is a set of ambiguity sets for source-action pair ``(s, a)``, where each ``\Gamma_{s, a} = \bigotimes_{i=1}^n \Gamma^i_{s, a}`` with ``\Gamma^i_{s, a}`` is an interval ambiguity set over the ``i``-th marginal, i.e. over ``S_i``.

An odIMDP is equivalent to an fRMDP where the dependency graph is ``\mathcal{G} = (\mathcal{V}, \mathcal{E})`` with ``\mathcal{V} = \{S_1, \ldots, S_n, A\} \cup \{S'_1, \ldots, S'_n\}`` and ``\mathcal{E} = \{(S_i, S'_j) : i, j = 1, \ldots, n\} \cup \{(A_i, S'_j) : j = 1, \ldots, m, i = 1, \ldots, n\}``. In other words, each next-state variable ``S'_i`` depends on all state and action variables and the dependency graph is a complete bipartite graph. Paths, strategies, and adversaries are defined similarly to fRMDPs.

## fIMDPs
Factored IMDPs (fIMDPs) are a subclass of fRMDPs where each marginal ambiguity set is an interval ambiguity set, but where the dependency graph can be arbitrary. 
Formally, an fIMDP ``M`` with ``n`` marginals is a tuple ``M = (S, S_0, A, \mathcal{G}, \Gamma)``, where

- ``S = S_1 \times \cdots \times S_n`` is a finite set of joint states with ``S_i`` being a finite set of states for the ``i``-th marginal,
- ``S_0 \subseteq S`` is a set of initial states,
- ``A`` is a finite set of actions,
- ``\mathcal{G} = (\mathcal{V}, \mathcal{E})`` is a directed bipartite graph with nodes ``\mathcal{V} = \mathcal{V}_{ind} \cup \mathcal{V}_{cond} = \{S_1, \ldots, S_n, A_1, \ldots, A_m\} \cup \{S'_1, \ldots, S'_n\}`` representing the state and action variables and their next-state counterparts, and edges ``\mathcal{E} \subseteq \mathcal{V}_{ind} \times \mathcal{V}_{cond}`` representing dependencies of ``S'_i`` on ``S_j`` and ``A_k``,
- ``\Gamma = \{\Gamma_{s, a}\}_{s \in S, a \in A}`` is a set of ambiguity sets for source-action pair ``(s, a)``, where each ``\Gamma_{s, a} = \bigotimes_{i=1}^n \Gamma^i_{\text{Pa}_\mathcal{G}(S'_i) \cap (s, a)}`` with ``\Gamma^i_{\text{Pa}_\mathcal{G}(S'_i) \cap (s, a)}`` is an interval ambiguity set over the ``i``-th marginal, i.e. over ``S_i``, conditional on the values in ``(s, a)`` of the parent variables ``\text{Pa}_\mathcal{G}(S'_i)`` of ``S'_i`` in ``\mathcal{G}``.

The example in [Factored RMDPs](@ref) is also an example of an fIMDP.

### References
```@bibliography
Pages = ["models.md"]
Canonical = false
```