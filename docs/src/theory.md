# Theory 
Notation: A probability distribution ``\gamma`` over a finite set ``S`` is a function ``\gamma : S \to [0, 1]`` satisfying ``\sum_{s \in S} \gamma(s) = 1``. We denote by ``\mathcal{D}(S)`` the set of all probability distributions over ``S``. 
For ``\underline{p}, \overline{p} : S \to [0, 1]`` such that ``\underline{p}(s) \leq \overline{p}(s)`` for each ``s \in S`` and ``\sum_{s \in S} \underline{p}(s) \leq 1 \leq \sum_{s \in S} \overline{p}(s)``, an interval ambiguity set ``\Gamma \subset \mathcal{D}(S)`` is the set of distributions such that 
```math
    \Gamma = \{ \gamma \in \mathcal{D}(S) \,:\, \underline{p}(s) \leq \gamma(s) \leq \overline{p}(s) \text{ for each } s\in S \}.
```
``\underline{p}, \overline{p}`` are referred to as the interval bounds of the interval ambiguity set.
For ``n`` finite sets ``S_1, \ldots, S_n`` we denote by ``S_1 \times \cdots \times S_n`` their Cartesian product. Given ``S=S_1 \times \cdots \times S_n`` and ``n`` ambiguity sets ``\Gamma_i \in \mathcal{D}(S_i)``, ``i = 1, \ldots, n``, the product ambiguity set ``\Gamma \subseteq \mathcal{D}(S)`` is defined as: 
```math
    \Gamma = \left\{ \gamma \in \mathcal{D}(S) \,:\, \gamma(s) = \prod_{i=1}^n \gamma^i(s^i), \, \gamma^i \in \Gamma_i \right\}
```
where ``s = (s_1, \ldots, s_n)\in S``. We will denote the product ambiguity set as ``\Gamma = \bigotimes_{i=1}^n \Gamma_i``. Each ``\Gamma_i`` is called a marginal or component ambiguity set.

## IMDPs
Interval Markov Decision Processes (IMDPs), also called bounded-parameter MDPs [1], are a generalization of MDPs, where the transition probabilities, given source state and action, are not known exactly, but they are constrained to be in some probability interval. 
Formally, an IMDP ``M`` is a tuple ``M = (S, S_0`, A, \Gamma)``, where

- ``S`` is a finite set of states,
- ``S_0 \subseteq S`` is a set of initial states,
- ``A`` is a finite set of actions,
- ```\Gamma = \{\Gamma_{s,a}\}_{s\in S,a \in A}`` is a set of ambiguity sets for source-action pair ``(s, a)``, where each ``\Gamma_{s,a}`` is an interval ambiguity set over ``S``.

A path of an IMDP is a sequence of states and actions ``\omega = (s_0,a_0),(s_1,a_1),\dots``, where ``(s_i,a_i)\in S \times A``. We denote by ``\omega(k) = s_k`` the state of the path at time ``k \in \mathbb{N}^0`` and by ``\Omega`` the set of all paths.  
A _strategy_ or _policy_ for an IMDP is a function ``\pi`` that assigns an action to a given state of an IMDP. _Time-dependent_ strategies are functions from state and time step to an action, i.e. ``\pi: S\times \mathbb{N}^0 \to A``. If ``\pi`` does not depend on time and solely depends on the current state, it is called a _stationary_ strategy. Similar to a strategy, an adversary ``\eta`` is a function that assigns a feasible distribution to a given state. Given a strategy and an adversary, an IMDP collapses to a finite Markov chain.

## OD-IMDPs
Orthogonally Decoupled IMDPs (OD-IMDPs) are a subclass of robust MDPs that are designed to be more memory-efficient and computationally efficient than the general IMDP model. The states are structured into an orthogonal, or grid-based, decomposition, and and the transition probability ambiguity sets, for each source-action pair (note the ``(s, a)``-rectangularity [2]), as a product of interval ambiguity sets along each marginal. 

Formally, an OD-IMDP ``M`` with ``n`` marginals is a tuple ``M = (S, S_0, A, \Gamma)``, where

- ``S = S_1 \times \cdots \times S_n`` is a finite set of joint states with ``S_i`` being a finite set of states for the ``i``-th marginal,
- ``S_0 \subseteq S`` is a set of initial states,
- ``A`` is a finite set of actions,
- ``\Gamma = \{\Gamma_{s,a}\}_{s\in S,a \in A}`` is a set of ambiguity sets for source-action pair ``(s, a)``, where each ``\Gamma_{s,a} = \bigotimes_{i=1}^n \Gamma^i_{s,a}`` with ``\Gamma^i_{s,a}`` is an interval ambiguity set over the ``i``-th marginal, i.e. over ``S_i``.

Paths, strategies, and adversaries are defined similarly to IMDPs. See [3] for more details on OD-IMDPs.

## Mixtures of OD-IMDPs
Mixtures of OD-IMDPs are included to address the issue the OD-IMDPs may not be able to represent all uncertainty in the transition probabilities. The mixture model is a convex combination of OD-IMDPs, where each OD-IMDP has its own set of ambiguity sets. Furthermore, the weights of the mixture are also interval-valued.

Formally, a mixture of OD-IMDPs ``M`` with ``K`` OD-IMDPs and ``n`` marginals is a tuple ``M = (S, S_0, A, \Gamma, \Gamma^\alpha)``, where
- ``S = S_1 \times \cdots \times S_n`` is a finite set of joint states with ``S_i`` being a finite set of states for the ``i``-th marginal,
- ``S_0 \subseteq S`` is a set of initial states,
- ``A`` is a finite set of actions,
- ``\Gamma = \{\Gamma_{r,s,a}\}_{r \in K, s\in S,a \in A}`` is a set of ambiguity sets for source-action pair ``(s, a)`` and OD-IMDP ``R``, where each ``\Gamma_{r,s,a} = \bigotimes_{i=1}^n \Gamma^i_{r,s,a}`` with ``\Gamma^i_{r,s,a}`` is an interval ambiguity set over the ``i``-th marginal, i.e. over ``S_i``.
- ``\Gamma^\alpha = \{\Gamma^\alpha_{s,a}\}_{s \in S, a \in A}`` is a set of interval ambiguity sets for the weights of the mixture, i.e. over ``\{1, \ldots, K\}``.

A feasible distribution for a mixture of OD-IMDPs is ``\sum_{r \in K} \alpha_{s,a}(r) \prod_{i = 1}^n \gamma_{r,s,a}`` where ``\alpha_{s,a} \in \Gamma^\alpha_{s,a}`` and ``\gamma_{r,s,a} \in \Gamma_{r,s,a}`` for each source-action pair ``(s, a)``. See [3] for more details on mixtures of OD-IMDPs.

### Reachability
In this formal framework, we can describe computing reachability given a target set ``G`` and a horizon ``K \in \mathbb{N} \cup \{\infty\}`` as the following objective 

```math
{\mathop{opt}\limits_{\pi}}^{\pi} \; {\mathop{opt}\limits_{\eta}}^{\eta} \; \mathbb{P}_{\pi,\eta }\left[\omega \in \Omega : \exists k \in [0,K], \, \omega(k)\in G  \right],
```

where ``\mathop{opt}^{\pi},\mathop{opt}^{\eta} \in \{\min, \max\}`` and ``\mathbb{P}_{\pi,\eta }`` is the probability of the Markov chain induced by strategy ``\pi`` and adversary ``\eta``.
When ``\mathop{opt}^{\eta} = \min``, the solution is called optimal _pessimistic_ probability (or reward), and conversely is called optimal _optimistic_ probability (or reward) when ``\mathop{opt}^{\eta} = \max``.
The choice of the min/max for the action and pessimistic/optimistic probability depends on the application. 

### Discounted reward
Discounted reward is similar to reachability but instead of a target set, we have a reward function ``r: S \to \mathbb{R}`` and a discount factor ``\gamma \in (0, 1)``. The objective is then

```math
{\mathop{opt}\limits_{\pi}}^{\pi} \; {\mathop{opt}\limits_{\eta}}^{\eta} \; \mathbb{E}_{\pi,\eta }\left[\sum_{k=0}^{K} \gamma^k r(\omega(k)) \right].
```

[1] Givan, Robert, Sonia Leach, and Thomas Dean. "Bounded-parameter Markov decision processes." Artificial Intelligence 122.1-2 (2000): 71-109.

[2] Suilen, M., Badings, T., Bovy, E. M., Parker, D., & Jansen, N. (2024). Robust Markov Decision Processes: A Place Where AI and Formal Methods Meet. In Principles of Verification: Cycling the Probabilistic Landscape: Essays Dedicated to Joost-Pieter Katoen on the Occasion of His 60th Birthday, Part III (pp. 126-154). Cham: Springer Nature Switzerland.

[3] Mathiesen, F. B., Haesaert, S., & Laurenti, L. (2024). Scalable control synthesis for stochastic systems via structural IMDP abstractions. arXiv preprint arXiv:2411.11803.