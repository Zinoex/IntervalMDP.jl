# Specifications
Specifications are compromised of a property and whether to minimize or maximize either the lower bound (pessimistic) or the upper bound (optimistic) ofthe value function. The property, or goal, e.g. reachability and reach-avoid, defines both how the value function is initialized and how it is updated after every Bellman iteration. The property also defines whether the horizon is finite or infinite, which impacts the stopping criteria and the resulting strategy type. In particular, for the infinite horizon, model checking algorithm continues until a convergence threshold is met and the strategy, if performing control synthesis, is stationary, while for a finite horizon, the strategy is time varying. 

!!! note
    The adversary is never synthesized directly and is always considered time-varying and dynamic. Over the infinite horizon, similar to the strategy, a time-varying adversary at convergence coincides with a stationary and static adversary [suilen2024robust](@cite). Without loss of generality below, we assume that the adversary ``\eta`` and strategy ``\pi`` are given.

As an example of constructing the specification, we consider here a reachability specification for an IMDP.
```@example
using IntervalMDP # hide

time_horizon = 10
prop = FiniteTimeReachability([3, 9, 10], time_horizon)

spec = Specification(prop)  # Default: Pessimistic, Maximize

# Explicit satisfaction mode (pessimistic/optimistic)
spec = Specification(prop, Pessimistic) # Default: Maximize, useful for Markov chains
spec = Specification(prop, Optimistic)

# Explicit strategy mode (minimize/maxize)
spec = Specification(prop, Pessimistic, Maximize)
spec = Specification(prop, Pessimistic, Minimize)  # Unusual, but available
spec = Specification(prop, Optimistic, Maximize)  # Unusual, but available
spec = Specification(prop, Optimistic, Minimize)
```


## Simple properties
In the sections below, we will enumerate the possible simple properties (meaning no task automaton required), their equivalence to some value function, and how to construct them. For complex properties and how to construct task automata see [Complex properties](@ref),

### Reachability
Given a target set ``G \subset S`` and a horizon ``K \in \mathbb{N} \cup \{\infty\}``, reachability is the following objective 
```math
\mathbb{P}^{\pi, \eta}_{\mathrm{reach}}(G, K) = \mathbb{P}^{\pi, \eta} \left[\omega \in \Omega : \exists k \in \{0, \ldots, K\}, \, \omega[k] \in G \right].
```

The property is equivalent to the following value function
```math
    \begin{aligned}
        V^{\pi, \eta}_0(s) &= \mathbf{1}_{G}(s)\\
        V^{\pi, \eta}_k(s) &= \mathbf{1}_{G}(s) + \mathbf{1}_{S \setminus G}(s) \mathbb{E}_{t \sim \eta(s, a, K - k)}[V^{\pi, \eta}_{k - 1}(t)]
    \end{aligned}
```
such that ``\mathbb{P}^{\pi, \eta}_{\mathrm{reach}}(G, K) = V_K(s)``, where for ``K = \infty`` the adversary does not depend on time.

Example:
```@example
using IntervalMDP # hide
# Finite horizon
time_horizon = 10

# Example with a single state variable
prop = FiniteTimeReachability([3, 9, 10], time_horizon)          # Single state variable only
prop = FiniteTimeReachability([(3,), (9,), (10,)], time_horizon) # Format available for multiple state variables

# Example with 3 state variables
prop = FiniteTimeReachability([(4, 3, 9)], time_horizon)

# Infinite horizon
convergence_threshold = 1e-8
prop = InfiniteTimeReachability([3, 9, 10], convergence_threshold)
```

In addition to finite and infinite horizon reachability, we also define _exact_ time reachability, which is the following property
```math
\mathbb{P}^{\pi, \eta}_{\mathrm{exact-reach}}(G, K) = \mathbb{P}^{\pi, \eta} \left[\omega \in \Omega : \omega[K] \in G \right],
```
which is equivalent with the following value function
```math
    \begin{aligned}
        V^{\pi, \eta}_0(s) &= \mathbf{1}_{G}(s)\\
        V^{\pi, \eta}_k(s) &= \mathbb{E}_{t \sim \eta(s, a, K - k)}[V^{\pi, \eta}_{k - 1}(t)]
    \end{aligned}
```
such that ``\mathbb{P}^{\pi, \eta}_{\mathrm{exact-reach}}(G, K) = V_K(s)`` for a horizon ``K \in \mathbb{N}``.

This can be constructed similarly
```@example
using IntervalMDP # hide
time_horizon = 10

# Example with a single state variable
prop = ExactTimeReachability([3, 9, 10], time_horizon)          # Single state variable only
prop = ExactTimeReachability([(3,), (9,), (10,)], time_horizon) # Format available for multiple state variables

# Example with 3 state variables
prop = ExactTimeReachability([(4, 3, 9)], time_horizon)
```

### Reach-avoid
Given a target set ``G \subset S``, an avoid set ``O \subset S`` (with ``G \cap O = \emptyset``), and a horizon ``K \in \mathbb{N} \cup \{\infty\}``, reach-avoid is the following objective 
```math
\mathbb{P}^{\pi, \eta}_{\mathrm{reach-avoid}}(G, O, K) = \mathbb{P}^{\pi, \eta} \left[\omega \in \Omega : \exists k \in \{0, \ldots, K\}, \, \omega[k] \in G, \; \forall k' \in \{0, \ldots, k' \}, \, \omega[k] \notin O \right].
```

The property is equivalent to the following value function
```math
    \begin{aligned}
        V^{\pi, \eta}_0(s) &= \mathbf{1}_{G}(s)\\
        V^{\pi, \eta}_k(s) &= \mathbf{1}_{G}(s) + \mathbf{1}_{S \setminus (G \cup O)}(s) \mathbb{E}_{t \sim \eta(s, a, K - k)}[V^{\pi, \eta}_{k - 1}(t)]
    \end{aligned}
```
such that ``\mathbb{P}^{\pi, \eta}_{\mathrm{reach-avoid}}(G, O, K) = V_K(s)``, where for ``K = \infty`` the adversary does not depend on time.

Example:
```@example
using IntervalMDP # hide
# Finite horizon
time_horizon = 10

# Example with a single state variable
reach = [3, 9]
avoid = [10]
prop = FiniteTimeReachAvoid(reach, avoid, time_horizon) # Single state variable only

reach = [(3,), (9,)]
avoid = [(10,)]
prop = FiniteTimeReachAvoid(reach, avoid, time_horizon) # Format available for multiple state variables

# Example with 3 state variables
reach = [(4, 3, 9)]
avoid = [(1, 1, 9)]
prop = FiniteTimeReachAvoid(reach, avoid, time_horizon)

# Infinite horizon
convergence_threshold = 1e-8
prop = InfiniteTimeReachAvoid(reach, avoid, convergence_threshold)
```

We also define _exact_ time reach-avoid, which is the following property
```math
\mathbb{P}^{\pi, \eta}_{\mathrm{exact-reach-avoid}}(G, O, K) = \mathbb{P}^{\pi, \eta} \left[\omega \in \Omega : \omega[K] \in G, \; \forall k \in \{0, \ldots, K\}, \, \omega[k] \notin O \right],
```
which is equivalent with the following value function
```math
    \begin{aligned}
        V^{\pi, \eta}_0(s) &= \mathbf{1}_{G}(s)\\
        V^{\pi, \eta}_k(s) &= \mathbf{1}_{S \setminus O}(s)\mathbb{E}_{t \sim \eta(s, a, K - k)}[V^{\pi, \eta}_{k - 1}(t)]
    \end{aligned}
```
such that ``\mathbb{P}^{\pi, \eta}_{\mathrm{exact-reach}}(G, K) = V_K(s)`` for a horizon ``K \in \mathbb{N}``.

This can be constructed similarly
```@example
using IntervalMDP # hide
time_horizon = 10

# Example with a single state variable
reach = [3, 9]
avoid = [10]
prop = ExactTimeReachAvoid(reach, avoid, time_horizon) # Single state variable only

reach = [(3,), (9,)]
avoid = [(10,)]
prop = ExactTimeReachAvoid(reach, avoid, time_horizon) # Format available for multiple state variables

# Example with 3 state variables
reach = [(4, 3, 9)]
avoid = [(1, 1, 9)]
prop = ExactTimeReachAvoid(reach, avoid, time_horizon)
```

### Safety
Given an avoid set ``O \subset S`` and a horizon ``K \in \mathbb{N} \cup \{\infty\}``, safety is the following objective 
```math
\mathbb{P}^{\pi, \eta}_{\mathrm{safe}}(O, K) = \mathbb{P}^{\pi, \eta} \left[\omega \in \Omega : \forall k \in \{0, \ldots, K\}, \, \omega[k] \notin O \right].
```
This property can by duality with reachability equivalently be states as ``\mathbb{P}^{\pi, \eta}_{\mathrm{safe}}(O, K) = 1 - \mathbb{P}^{\pi, \eta}_{\mathrm{reach}}(O, K)``. Note that if the strategy and adversary are not given, their optimization direction must be flipped in the dual objective. Alternatively, the property can be stated via the following value function
```math
    \begin{aligned}
        V^{\pi, \eta}_0(s) &= -\mathbf{1}_{O}(s)\\
        V^{\pi, \eta}_k(s) &= -\mathbf{1}_{O}(s) + \mathbf{1}_{S \setminus O}(s) \mathbb{E}_{t \sim \eta(s, a, K - k)}[V^{\pi, \eta}_{k - 1}(t)]
    \end{aligned}
```
such that ``\mathbb{P}^{\pi, \eta}_{\mathrm{safe}}(G, K) = 1 + V_K(s)``, where for ``K = \infty`` the adversary does not depend on time.
The benefit of this formulation is that the optimization directions need not be flipped.

Example:
```@example
using IntervalMDP # hide
# Finite horizon
time_horizon = 10

# Example with a single state variable
prop = FiniteTimeSafety([10], time_horizon)    # Single state variable only
prop = FiniteTimeSafety([(10,)], time_horizon) # Format available for multiple state variables

# Example with 3 state variables
prop = FiniteTimeSafety([(4, 3, 9)], time_horizon)

# Infinite horizon
convergence_threshold = 1e-8
prop = InfiniteTimeSafety([3, 9, 10], convergence_threshold)
```

### Discounted reward
Given a (state) reward function ``r : S \to \mathbb{R}``, a discount factor ``\nu \in (0, 1)``, and a horizon ``K \in \mathbb{N} \cup \{\infty\}``, a (discounted) reward objective is then follow
```math
\mathbb{E}^{\pi,\eta}_{\mathrm{reward}}(r, \nu, K) = \mathbb{E}^{\pi,\eta}\left[\sum_{k=0}^{K} \nu^k r(\omega[k]) \right].
```
For a finite horizon, the discount factor is allowed to be ``\nu = 1``; for the infinite horizon, ``\nu < 1`` is required for convergence.

The property is equivalent to the following value function
```math
    \begin{aligned}
        V^{\pi, \eta}_0(s) &= r(s)\\
        V^{\pi, \eta}_k(s) &= r(s) + \nu \mathbb{E}_{t \sim \eta(s, a, K - k)}[V^{\pi, \eta}_{k - 1}(t)]
    \end{aligned}
```
such that ``\mathbb{E}^{\pi,\eta}_{\mathrm{reward}}(r, \nu, K) = V_K(s)``, where for ``K = \infty`` the adversary does not depend on time.

Example:
```@example
using IntervalMDP # hide
# Finite horizon
time_horizon = 10
discount_factor = 0.9

# Example with a single state variable
rewards = [0.0, 2.0, 1.0, -1.0]  # For 4 states
prop = FiniteTimeReward(rewards, discount_factor, time_horizon)

# Example with 2 state variables of 2 and 4 values respectively
rewards = [
    0.0  2.0  1.0 -1.0;
    1.0 -1.0  0.0  2.0
]
prop = FiniteTimeReward(rewards, discount_factor, time_horizon)

# Infinite horizon
convergence_threshold = 1e-8
prop = InfiniteTimeReward(rewards, discount_factor, convergence_threshold)
```

### Expected exit time
Given a avoid set ``O \subset S``, the expected exit time of the set `S \setminus O` is the following objective 
```math
\mathbb{E}^{\pi,\eta}_{\mathrm{exit}}(O) = \mathbb{E}^{\pi,\eta}\left[k : \omega[k] \in O, \, \forall k' \in \{0, \ldots, k\}, \, \omega[k'] \notin O \right].
```

The property is equivalent to the following value function
```math
    \begin{aligned}
        V^{\pi, \eta}_0(s) &= \mathbf{1}_{S \setminus 0}(s)\\
        V^{\pi, \eta}_k(s) &= \mathbf{1}_{S \setminus O}(s) \left(1 + \mathbb{E}_{t \sim \eta(s, a)}[V^{\pi, \eta}_{k - 1}(t)]\right)
    \end{aligned}
```
such that ``\mathbb{E}^{\pi,\eta}_{\mathrm{exit}}(O) = V_\infty(s)``. The adversary does not depend on time.

Example:
```@example
using IntervalMDP # hide

convergence_threshold = 1e-8

# Example with a single state variable
avoid_states = [10]
prop = ExpectedExitTime(avoid_states, convergence_threshold)    # Single state variable only

avoid_states = [(10,)]
prop = ExpectedExitTime(avoid_states, convergence_threshold) # Format available for multiple state variables

# Example with 3 state variables
avoid_states = [(4, 3, 9)]
prop = ExpectedExitTime(avoid_states, convergence_threshold)
```

## Complex properties
For complex, temporal properties, it is necessary to use some form of automaton to express the property. In this package, we support specifications via Deterministic Finite Automata (DFA), which via a lazy product construction with an fRMDP allows for efficient implementations of the Bellman operator. DFAs is an important class of task automata as it can express properties in syntactically co-safe Linear Temporal Logic (scLTL) [baier2008principles](@cite) and Linear Temporal Logic over finite traces (LTLf) [de2013linear](@cite).

Formally, a DFA is a tuple ``\mathcal{A} = (Q, q_0, 2^{\mathrm{AP}}, \delta, F)`` where ``Q`` is a finite set of states, ``q_0 \in Q`` is the initial state, ``2^{\mathrm{AP}}`` is a finite alphabet from atomic proposition ``\mathrm{AP}``, ``\delta : Q \times 2^{\mathrm{AP}} \to Q`` is a transition function, and ``F \subseteq Q`` is a set of accepting states. The DFA accepts a word ``\sigma = \sigma_0 \sigma_1 \ldots \sigma_n`` over the alphabet ``2^{\mathrm{AP}}`` if there exists a sequence of states ``q_0, q_1, \ldots q_n`` such that ``q_{i+1} = \delta(q_i, \sigma_i)`` for all ``0 \geq i < n`` and ``q_n \in F``. We write ``\mathcal{A} \models \sigma`` if the word ``\sigma`` is accepted by the DFA ``\mathcal{A}``.

A DFA can be constructed like in the following example[^1]:
```@example
using IntervalMDP # hide

atomic_props = ["a", "b"]

delta = TransitionFunction([  # Columns: states, rows: input symbols
    1 3 3  # symbol: ""
    2 1 3  # symbol: "a"
    3 3 3  # symbol: "b"
    1 1 1  # symbol: "ab"
])

initial_state = 1

dfa = DFA(delta, initial_state, atomic_props)
```
Notice that the DFA does not include the set of accepting states. This is because the accepting states does not impact the Bellman operator and therefore are defined in `DFAReachability` objects, which is shown below.

```@example
using IntervalMDP # hide

accepting_states = [3]  # Accepting _DFA_ states

time_horizon = 10
prop = FiniteTimeDFAReachability(accepting_states, time_horizon)

convergence_threshold = 1e-8
prop = InfiniteTimeDFAReachability(accepting_states, convergence_threshold)
```

Given an fRMDP ``M = (S, S_0, A, \mathcal{G}, \Gamma)`` and a labeling function ``L : S \to \Sigma`` that maps states of the fRMDP to symbols in the alphabet of the DFA, a path ``\omega = s_0 s_1 \ldots`` in the fRMDP produces a word ``L(s_0) L(s_1) \ldots`` that is (possibly) accepted by the DFA. The probability of producing a path in the fRMDP that is accepted by the DFA can be expressed via the product construction ``M \otimes \mathcal{A} = (Z, Z_0, A, \Gamma')``, where
- ``Z = S \times Q`` is the set of product states, 
- ``Z_0 = S_0 \times \{q_0\}`` is the set of initial product states,
- ``A`` is the set of actions, and
- ``\Gamma' = \{\Gamma_{z, a}\}_{z \in Z, a \in A}`` is the joint ambiguity set defined as
```math
\Gamma'_{z, a} = \{\gamma'_{z, a} \in \mathcal{D}(Z) : \exists \gamma_{s, a} \in \Gamma_{s, a} \text{ s.t. } \gamma'_{z, a}(z') = \mathbf{1}_{q'}(\delta(q, L(t))) \gamma_{s, a}(t)\}
```
where ``z = (s, q)`` and ``z' = (t, q')``. Then, the probability of generating a path, of length ``K \in \mathbb{N}``, in the fRMDP that is accepted by the DFA is formally defined as
```math
\mathbb{P}^{\pi, \eta}_{\mathrm{dfa-reach}}(F, K) = \mathbb{P}^{\pi, \eta}_{M \otimes \mathcal{A}} \left[\omega \in \Omega : \omega[K] \in S \times F \right].
```
Note that this is equivalent to reachability in the product fRMDP ``M \otimes \mathcal{A}``. Therefore, the property can equivalently be stated via the value function for reachability in the product fRMDP.
```math
    \begin{aligned}
        V^{\pi, \eta}_0(z) &= \mathbf{1}_{S \times F}(z)\\
        V^{\pi, \eta}_k(z) &= \mathbf{1}_{S \times F}(z) + \mathbf{1}_{Z \setminus (S \times F)}(z) \mathbb{E}_{z' \sim \eta(z, a, K - k)}[V^{\pi, \eta}_{k - 1}(z')]
    \end{aligned}
```
such that ``\mathbb{P}^{\pi, \eta}_{\mathrm{dfa-reach}}(F, K) = V_K(z)``. 

Note that the product is never explicitly constructed, for three reasons: (i) the result is an RMDP and not an fRMDP, thus negating the computational benefits of using fRMDPs, (ii) the transition function will be sparse even if some marginals in the original fRMDP are dense, and (iii) the Bellman operator will not be able to leverage the structure of the product construction. Instead, we lazily construct the product as a [`ProductProcess`](@ref), and sequentially update the value function first updating wrt. the DFA transition and then wrt. the fRMDP transition like
```math
    \begin{aligned}
        V^{\pi, \eta}_0(s, q) &= \mathbf{1}_{F}(q)\\
        W^{\pi, \eta}_k(t, q) &= \mathbf{1}_{F}(q) + \mathbf{1}_{Q \setminus F}(q) V^{\pi, \eta}_{k - 1}(t, \delta(q, L(t)))\\
        V^{\pi, \eta}_k(s, q) &= \mathbb{E}_{t \sim \eta(s, a, K - k)}[W^{\pi, \eta}_k(t, q)]
    \end{aligned}
```
Notice that ``W^{\pi, \eta}_k(t, q)`` is shared for all ``s \in S`` when updating ``V^{\pi, \eta}_k(s, q)``. This allows for efficient, cache-friendly implementations of the Bellman operator. The kernel for product processes merely forwards, for each DFA state ``q \in Q \setminus F``, the Bellman update to the underlying Bellman operator algorithm, which is chosen based on the fRMDP model type, e.g. IMDP or odIMDP, storage type, e.g. dense or sparse, and hardware, e.g. CPU or CUDA, for efficicency.

Example of constructing a product process:
```@setup product_process_example
using IntervalMDP

# Construct DFA
atomic_props = ["a", "b"]

delta = TransitionFunction([  # Columns: states, rows: input symbols
    1 3 3  # symbol: ""
    2 1 3  # symbol: "a"
    3 3 3  # symbol: "b"
    1 1 1  # symbol: "ab"
])

initial_state = 1

dfa = DFA(delta, initial_state, atomic_props)

# Construct fRMDP
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

transition_probs = [prob1, prob2]
istates = [Int32(1)]

mdp = IntervalMarkovDecisionProcess(transition_probs, istates)
```

```@example product_process_example
map = [1, 2, 3]  # "", "a", "b"
lf = LabellingFunction(map)

product_process = ProductProcess(mdp, dfa, lf)
```
The product process can then be used in a [`VerificationProblem`](@ref) or [`ControlSynthesisProblem`](@ref) together with a specification with a DFA property.

[^1]: The automatic construction of a DFA from scLTL or LTLf formulae is not currently supported, but planned for future releases.