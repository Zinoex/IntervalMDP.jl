# Algorithms

## Model checking
The core algorithmic component of this package is (robust) value iteration, which is used to solve verification and control synthesis problems for fRMDPs. Value iteration is an iterative algorithm that computes the value function for a given specification by repeatedly applying the [Bellman operator](@ref "Bellman operator algorithms") until convergence.

To simplify the dicussion on the algorithmic choices, we will assume that the goal is to compute the maximizing pessimistic probability of reaching a set of states ``G``, that is, 
```math
\max_{\pi} \; \min_{\eta} \;  \mathbb{P}^{\pi, \eta}_{\mathrm{reach}}(G, K).
```

See [Models](@ref) for more details on the formal definition of fRMDPs, strategies, and adversaries; in this case the maximization and minimization operators respectively. The algorithms are easily adapted to [other specifications](@ref "Specifications").

Computing the solution to the above problem can be framed in terms of value iteration. The value function ``V_k`` is the probability of reaching ``G`` in ``k`` steps or fewer. The value function is initialized to ``V_0(s) = 1`` if ``s \in G`` and ``V_0(s) = 0`` otherwise. The value function is then iteratively updated according to the Bellman equation
```math
\begin{aligned}
    V_{0}(s) &= \mathbf{1}_{G}(s) \\
    V_{k}(s) &= \mathbf{1}_{G}(s) + \mathbf{1}_{S \setminus G}(s) \max_{a \in A} \min_{\gamma_{s,a} \in \Gamma_{s,a}} \sum_{t \in S} V_{k-1}(t) \gamma_{s,a}(t),
\end{aligned}
```
where ``\mathbf{1}_{G}(s) = 1``  if ``s \in G`` and ``0`` otherwise is the indicator function for set ``G``. This Bellman update is repeated until ``k = K``, or if ``K = \infty``, until the value function converges, i.e. ``V_k = V_{k-1}`` for some ``k``. The value function is then the solution to the problem.

In a more programmatic formulation, the algorithm can be summarized as follows:
```julia
function value_iteration(system, spec)
    V = initialize_value_function(spec)  # E.g. V[s] = 1 if s in G else 0 for reachability

    while !converged(V)  # or for k in 1:K if K is finite
        # Compute max_{a \in A} \min_{γ_{s,a} \in Γ_{s,a}} \sum_{t \in S} V_{k-1}(t) γ_{s,a}(t) for all states s
        V = bellman_update(V, system) # System contains information about S, A, and Γ
        post_process!(V, spec) # E.g. set V[s] = 1 for s in G for reachability
    end
end
```
We slightly abuse terminology and call the max/min expectation the Bellman update, even though it is not a proper Bellman operator as it does not include the indicator function for ``G``. The min/max expectation is however shared between all specifications, and thus it is natural to separate it from the specification-dependent post-processing step.

Note that exact convergence is virtually, impossible, unless using (computationally slow) exact arithmetic, to achieve in a finite number of iterations due to the finite precision of floating point numbers. Hence, we instead use a residual tolerance ``\epsilon`` and stop when Bellman residual ``V_k - V_{k-1}`` is less than the threshold, ``\|V_k - V_{k-1}\|_\infty < \epsilon``. See [Bellman operator algorithms](@ref) for algorithms that support exact arithmetic.

## Bellman operator algorithms
As the Bellman update is the most computationally intensive part of the algorithm, it is crucial to implement it efficiently including considerations about type stability, pre-allocation and in-place operations, memory access patterns, and parallelization.

1. Type stability: the Bellman update should be type stable, i.e. the correct kernel to dispatch to should be inferable at compile time, to avoid dynamic dispatch and heap allocations in the hot loop. This can be achieved by using parametric types and avoiding abstract types in the hot loop.
2. Pre-allocation and in-place operations: to avoid unnecessary allocations and reducing GC pressure, the value function (pre and post Bellman update) is be pre-allocated and updated in-place, and the Bellman update relies on pre-allocated workspace objects.
3. Memory access patterns: to ensure cache efficiency, the memory access pattern should be as contiguous as possible. This is achieved by storing the transition matrices/ambiguity sets in column-major order, where each column corresponds to a source-action pair.
4. Parallelization: to leverage multi-core CPUs and CUDA hardware, the Bellman update should be parallelized across source-states and in the case of CUDA, also across actions and target states.

A challenge with designing Bellman operator algorithms for fRMDPs is that ``\min_{\gamma_{s,a} \in \Gamma_{s,a}} \sum_{t \in S} V_{k-1}(t) \gamma_{s,a}(t)`` is not always computable exactly, and thus, we must resort to sound approximations. For IMDPs, the minimum can be computed exactly via [O-maximization](@ref). Below, we will describe different algorithms for computing the Bellman update, their trade-offs, and algorithmic choices for an efficient implementation.

### O-maximization
In case of an IMDP, the minimum over all feasible distributions can be computed as a solution to a Linear Programming (LP) problem, namely
```math
    \begin{aligned}
        \min_{\gamma_{s, a}} \quad & \sum_{t \in S} V_{k-1}(t) \cdot \gamma_{s, a}(t), \\
        \quad & \underline{\gamma}_{s, a}(t) \leq \gamma_{s, a}(t) \leq \overline{\gamma}_{s, a}(t) \quad \forall t \in S, \\
        \quad & \sum_{t \in S} \gamma_{s,a}(t) = 1.
    \end{aligned}
```
However, due to the particular structure of the LP problem, we can use a more efficient algorithm: O-maximization, or ordering-maximization [givan2000bounded, lahijanian2015formal](@cite).
In the case of pessimistic probability, we want to assign the most possible probability mass to the destinations with the smallest value of ``V_{k-1}``, while obeying that the probability distribution is feasible, i.e. within the probability bounds and that it sums to 1. This is done by sorting the values of ``V_{k-1}`` and then assigning state with the smallest value its upper bound, then the second smallest, and so on until the remaining mass must be assigned to the lower bound of the remaining states for probability distribution is feasible.
```julia
function min_value(V, system, source, action)
    # Sort values of `V` in ascending order
    order = sortperm(V)

    # Initialize distribution to lower bounds
    p = lower_bounds(system, source, action)
    budget = 1 - sum(p)

    # Assign upper bounds to states with smallest values
    # until remaining mass is zero
    for idx in order
        gap = upper_bounds(system, source, action)[idx] - p[idx]
        if budget <= gap
            p[idx] += budget
            break
        else
            p[idx] += gap
            budget -= gap
        end
    end

    v = dot(V, p)
    return v
end
```

For fIMDPs, O-maximization can be applied recursively over the marginals as a sound under-approximation of the minimum [mathiesen2025scalable](@cite). Let ``S = S_1 \times \cdots \times S_n`` be the state space factored into ``n`` state variables, and let ``\Gamma_{s,a} = \Gamma^1_{s,a} \times \cdots \times \Gamma^n_{s,a}`` be the transition ambiguity sets factored into ``n`` marginals. Then, we can compute a bound on the minimum as
```math
    \begin{aligned}
        W_{s,a}^{k,n}(t^1, \ldots, t^n) &= V_{k - 1}(t)\\
        W_{s,a}^{k,i-1}(t^1, \ldots, t^{i-1}) &= \min_{\gamma^i_{s,a} \in \Gamma^i_{s,a}} \sum_{t^i \in S_i} W_{s,a}^{k,i}(
            t^1, \ldots, t^i) \gamma^i_{s,a}(t^i),\\
            &\qquad \qquad \text{ for } i = 2, \ldots, n \\
        W_{s,a}^{k} :=  W_{s,a}^{k,0} &= \min_{\gamma^1_{s,a} \in \Gamma^1_{s,a}} \sum_{t^1 \in S_1} W_{s,a}^{k,1}(t^1) \gamma^1_{s,a}(t^1).
    \end{aligned}
```
Then, ``V_k(s) := \mathbf{1}_{G}(s) + \mathbf{1}_{S \setminus G}(s) \max_{a \in A} W_{s,a}^{k}``. Note that this is strictly better than building a joint ambiguity set by multiplying the marginal interval bounds [mathiesen2025scalable](@cite).

The algorithm is the default Bellman algorithm for IMDPs, but not for fIMDPs. To explicitly select (recursive) O-maximization, do the following:
```@setup explicit_omax
using IntervalMDP
N = Float64

prob1 = IntervalAmbiguitySets(;
    lower = N[
        0     1//2
        1//10 3//10
        1//5  1//10
    ],
    upper = N[
        1//2  7//10
        3//5  1//2
        7//10 3//10
    ],
)

prob2 = IntervalAmbiguitySets(;
    lower = N[
        1//10 1//5
        1//5  3//10
        3//10 2//5
    ],
    upper = N[
        3//5 3//5
        1//2 1//2
        2//5 2//5
    ],
)

prob3 = IntervalAmbiguitySets(;
    lower = N[
        0 0
        0 0
        1 1
    ],
    upper = N[
        0 0
        0 0
        1 1
    ]
)

transition_probs = [prob1, prob2, prob3]

mdp = IntervalMarkovDecisionProcess(transition_probs)
prop = FiniteTimeReachability([3], 10)  # Reach state 3 within 10 timesteps
spec = Specification(prop, Pessimistic, Maximize)
problem = VerificationProblem(mdp, spec)
```
```@example explicit_omax
alg = RobustValueIteration(OMaximization())
result = solve(problem, alg)
nothing # hide
```
O-maximization supports both floating point and exact arithmetic, and it is implemented for both CPU and CUDA hardware.

### Vertex enumeration
A way to compute the minimum exactly for fIMDPs, and in general polytopic ambiguity sets, is via vertex enumeration [schnitzer2025efficient](@cite). The idea is to enumerate the Cartesian product of all vertices of each polytope and then compute the minimum over the vertices. This is however only feasible for few state values along each marginal, as the potential number of vertices for each marginal can grow with the factorial of the number of state values, and exponentially in the number of dimensions. Hence, this algorithm is only feasible for small problems, but it is included for completeness and as a reference implementation. To use vertex enumeration, do the following:
```@setup explicit_vertex
using IntervalMDP
N = Float64

state_vars = (2, 3)
action_vars = (1, 2)

marginal1 = Marginal(IntervalAmbiguitySets(;
    lower = N[
        1//15  7//30  1//15  13//30  4//15  1//6
        2//5   7//30  1//30  11//30  2//15  1//10
    ],
    upper = N[
        17//30   7//10  2//3   4//5  7//10   2//3
        9//10  13//15  9//10  5//6  4//5   14//15
    ]
), (1, 2), (1,), (2, 3), (1,))

marginal2 = Marginal(IntervalAmbiguitySets(;
    lower = N[
        1//30  1//3   1//6   1//15  2//5   2//15
        4//15  1//4   1//6   1//30  2//15  1//30
        2//15  7//30  1//10  7//30  7//15  1//5
    ],
    upper = N[
        2//3   7//15   4//5   11//30  19//30   1//2
        23//30  4//5   23//30   3//5    7//10   8//15
        7//15  4//5   23//30   7//10   7//15  23//30
    ]
), (2,), (2,), (3,), (2,))

mdp = FactoredRobustMarkovDecisionProcess(state_vars, action_vars, (marginal1, marginal2))

prop = FiniteTimeReachability([(2, 3)], 10)  # Reach state (2, 3) within 10 timesteps
spec = Specification(prop, Pessimistic, Maximize)
problem = VerificationProblem(mdp, spec)
```
```@example explicit_vertex
alg = RobustValueIteration(VertexEnumeration())
result = solve(problem, alg)
nothing # hide
```
The implementation iterates vertex combinations in a lazy manner, and thus, it does not store all vertices in memory. Furthermore, efficient generation of vertices for each marginal is done via backtracking to avoid enumerating all possible orderings.

Vertex enumeration supports both floating point and exact arithmetic.

### Recursive McCormick envelopes
Another method for computing a sound under-approximation of the minimum for fIMDPs is via recursive McCormick envelopes [schnitzer2025efficient](@cite). The idea is to relace each bilinear term ``\gamma^1_{s, a}(t^1) \cdot \gamma^2_{s, a}(t^2)`` in ``\sum_{t \in S} V_{k-1}(') \gamma^1_{s, a}(t^1) \cdot \gamma^2_{s, a}(t^2)`` (for a system with two marginals) with a new variable ``q_{s, a}(t^1, t^2)`` and add linear McCormick constraints to ensure that ``q_{s, a}(t^1, t^2)`` is an over-approximation of the bilinear term. That is,
```math
    \begin{aligned}
        q_{s, a}(t^1, t^2) &\geq \underline{\gamma}^1_{s,a}(t^1) \cdot \gamma^2_{s,a}(t^2) + \underline{\gamma}^2_{s,a}(t^2) \cdot \gamma^1_{s,a}(t^1) - \underline{\gamma}^1_{s,a}(t^1) \cdot \underline{\gamma}^2_{s,a}(t^2), \\
        q_{s, a}(t^1, t^2) &\geq \overline{\gamma}^1_{s,a}(t^1) \cdot \gamma^2_{s,a}(t^2) + \overline{\gamma}^2_{s,a}(t^2) \cdot \gamma^1_{s,a}(t^1) - \overline{\gamma}^1_{s,a}(t^1) \cdot \overline{\gamma}^2_{s,a}(t^2), \\
        q_{s, a}(t^1, t^2) &\leq \underline{\gamma}^1_{s,a}(t^1) \cdot \gamma^2_{s,a}(t^2) + \overline{\gamma}^2_{s,a}(t^2) \cdot \gamma^1_{s,a}(t^1) - \underline{\gamma}^1_{s,a}(t^1) \cdot \overline{\gamma}^2_{s,a}(t^2), \\
        q_{s, a}(t^1, t^2) &\leq \overline{\gamma}^1_{s,a}(t^1) \cdot \gamma^2_{s,a}(t^2) + \underline{\gamma}^2_{s,a}(t^2) \cdot \gamma^1_{s,a}(t^1) - \overline{\gamma}^1_{s,a}(t^1) \cdot \underline{\gamma}^2_{s,a}(t^2).
    \end{aligned}
```
In addition, we add the constraint that ``\sum_{t^1 \in S_1} \sum_{t^2 \in S_2} q_{s, a}(t^1, t^2) = 1`` such that ``q_{s, a}`` is a valid probability distribution.

This results in a Linear Programming (LP) problem that can be solved efficiently. The McCormick envelopes can be applied recursively for more than two marginals. The algorithm is more efficient than vertex enumeration and is thus the default Bellman algorithm for fIMDPs.

To use recursive McCormick envelopes, do the following:
```@setup explicit_mccormick
using IntervalMDP, HiGHS
N = Float64

state_vars = (2, 3)
action_vars = (1, 2)

marginal1 = Marginal(IntervalAmbiguitySets(;
    lower = N[
        1//15  7//30  1//15  13//30  4//15  1//6
        2//5   7//30  1//30  11//30  2//15  1//10
    ],
    upper = N[
        17//30   7//10  2//3   4//5  7//10   2//3
        9//10  13//15  9//10  5//6  4//5   14//15
    ]
), (1, 2), (1,), (2, 3), (1,))

marginal2 = Marginal(IntervalAmbiguitySets(;
    lower = N[
        1//30  1//3   1//6   1//15  2//5   2//15
        4//15  1//4   1//6   1//30  2//15  1//30
        2//15  7//30  1//10  7//30  7//15  1//5
    ],
    upper = N[
        2//3   7//15   4//5   11//30  19//30   1//2
        23//30  4//5   23//30   3//5    7//10   8//15
        7//15  4//5   23//30   7//10   7//15  23//30
    ]
), (2,), (2,), (3,), (2,))

mdp = FactoredRobustMarkovDecisionProcess(state_vars, action_vars, (marginal1, marginal2))
prop = FiniteTimeReachability([(2, 3)], 10)  # Reach state (2, 3) within 10 timesteps
spec = Specification(prop, Pessimistic, Maximize)
problem = VerificationProblem(mdp, spec)
```
```@example explicit_mccormick
# Use default LP solver (HiGHS)
alg = RobustValueIteration(LPMcCormickRelaxation())

# Choose a different LP solver
using Clarabel
alg = RobustValueIteration(LPMcCormickRelaxation(; lp_solver=Clarabel.Optimizer))

result = solve(problem, alg)
nothing # hide
```

See the [JuMP documentation](https://jump.dev/JuMP.jl/stable/installation/#Supported-solvers) for a list of supported LP solvers. The recursive McCormick envelopes Bellman operator algorithm supports primarily floating point, but also exact arithmetic if the chosen LP solver does.