# Specifications
Specifications are compromised of a property and whether to minimize or maximize either the lower bound (pessimistic) or the upper bound (optimistic) ofthe value function. The property, or goal, e.g. reachability and reach-avoid, defines both how the value function is initialized and how it is updated after every Bellman iteration. The property also defines whether the horizon is finite or infinite, which impacts the stopping criteria and the resulting strategy type. In particular, for the infinite horizon, model checking algorithm continues until a convergence threshold is met and the strategy, if performing control synthesis, is stationary, while for a finite horizon, the strategy is time varying. 

!!! note
    The adversary is never synthesized directly and is always considered time-varying and dynamic. Over the infinite horizon, similar to the strategy, a time-varying adversary at convergence coincides with a stationary and static adversary (CITE). Without loss of generality below, we assume that the adversary ``\eta`` and strategy ``\pi`` are given.

!!! todo
    Add the proper citation everywhere

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
spec = Specification(prop, Optimistic, Maximize)  # Unusual, but avialable
spec = Specification(prop, Optimistic, Minimize)
```


## Simple properties
In the sections below, we will enumerate the possible simple properties (meaning no task automaton required), their equivalence to some value function, and how to construct them. For complex properties and how to construct task automata see [Complex properties](@ref),

### Reachability
Given a target set ``G`` and a horizon ``K \in \mathbb{N} \cup \{\infty\}`` reachability is the following objective 
```math
\mathbb{P}^{\pi, \eta}_{\mathrm{reach}}(G, K) = \mathbb{P}^{\pi, \eta} \left[\omega \in \Omega : \exists k \in \{0, \ldots, K\}, \, \omega[k] \in G \right].
```

The property is equivalent to the following value function
```math
    \begin{aligned}
        V^{\pi, \eta}_0(s) &= \mathbf{1}_{G}(s)\\
        V^{\pi, \eta}_k(s) &= \mathbf{1}_{G}(s) + \mathbf{1}_{S \setminus G}(s) \mathbb{E}_{s' \sim \eta(s, a, K - k)}[V_{k + 1}(s')]
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
        V^{\pi, \eta}_k(s) &= \mathbb{E}_{s' \sim \eta(s, a, K - k)}[V_{k + 1}(s')]
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
Given a target set ``G``, an avoid set ``O`` (with ``G \cap O = \emptyset``), and a horizon ``K \in \mathbb{N} \cup \{\infty\}`` reach-avoid is the following objective 
```math
\mathbb{P}^{\pi, \eta}_{\mathrm{reach-avoid}}(G, O, K) = \mathbb{P}^{\pi, \eta} \left[\omega \in \Omega : \exists k \in \{0, \ldots, K\}, \, \omega[k] \in G, \; \forall k' \in \{0, \ldots, k' \}, \, \omega[k] \notin O \right].
```

The property is equivalent to the following value function
```math
    \begin{aligned}
        V^{\pi, \eta}_0(s) &= \mathbf{1}_{G}(s)\\
        V^{\pi, \eta}_k(s) &= \mathbf{1}_{G}(s) + \mathbf{1}_{S \setminus (G \cup O)}(s) \mathbb{E}_{s' \sim \eta(s, a, K - k)}[V_{k + 1}(s')]
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
        V^{\pi, \eta}_k(s) &= \mathbf{1}_{S \setminus O}(s)\mathbb{E}_{s' \sim \eta(s, a, K - k)}[V_{k + 1}(s')]
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
Given an avoid set ``O`` and a horizon ``K \in \mathbb{N} \cup \{\infty\}`` safety is the following objective 
```math
\mathbb{P}^{\pi, \eta}_{\mathrm{safe}}(O, K) = \mathbb{P}^{\pi, \eta} \left[\omega \in \Omega : \forall k \in \{0, \ldots, K\}, \, \omega[k] \notin O \right].
```
This property can by duality with reachability equivalently be states as ``\mathbb{P}^{\pi, \eta}_{\mathrm{safe}}(O, K) = 1 - \mathbb{P}^{\pi, \eta}_{\mathrm{reach}}(G, K)``. Note that if the strategy and adversary are not given, their optimization direction must be flipped in the dual objective. Alternatively, the property can be stated via the following value function
```math
    \begin{aligned}
        V^{\pi, \eta}_0(s) &= -\mathbf{1}_{O}(s)\\
        V^{\pi, \eta}_k(s) &= -\mathbf{1}_{O}(s) + \mathbf{1}_{S \setminus O}(s) \mathbb{E}_{s' \sim \eta(s, a, K - k)}[V_{k + 1}(s')]
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
Discounted reward is similar to reachability but instead of a target set, we have a reward function ``r: S \to \mathbb{R}`` and a discount factor ``\gamma \in (0, 1)``. The objective is then

```math
{\mathop{opt}\limits_{\pi}}^{\pi} \; {\mathop{opt}\limits_{\eta}}^{\eta} \; \mathbb{E}_{\pi,\eta }\left[\sum_{k=0}^{K} \gamma^k r(\omega(k)) \right].
```

## Complex properties
