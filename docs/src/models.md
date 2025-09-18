# Models 
#### Mathematical Notation 
We denote the natural numbers by ``\mathbb{N}`` and ``\mathbb{N}_0 = \mathbb{N} \cup \{0\}``. A probability distribution ``\gamma`` over a finite set ``S`` is a function ``\gamma : S \to [0, 1]`` satisfying ``\sum_{s \in S} \gamma(s) = 1``. We denote by ``\mathcal{D}(S)`` the set of all probability distributions over ``S``. 
For ``\underline{\gamma}, \overline{\gamma} : S \to [0, 1]`` such that ``\underline{\gamma}(s) \leq \overline{\gamma}(s)`` for each ``s \in S`` and ``\sum_{s \in S} \underline{\gamma}(s) \leq 1 \leq \sum_{s \in S} \overline{\gamma}(s)``, an interval ambiguity set ``\Gamma \subset \mathcal{D}(S)`` is the set of distributions such that 
```math
    \Gamma = \{ \gamma \in \mathcal{D}(S) \,:\, \underline{\gamma}(s) \leq \gamma(s) \leq \overline{\gamma}(s) \text{ for each } s\in S \}.
```
``\underline{\gamma}, \overline{\gamma}`` are referred to as the interval bounds of the interval ambiguity set.
For ``n`` finite sets ``S_1, \ldots, S_n`` we denote by ``S_1 \times \cdots \times S_n`` their Cartesian product. Given ``S = S_1 \times \cdots \times S_n`` and ``n`` ambiguity sets ``\Gamma_i \in \mathcal{D}(S_i)``, ``i = 1, \ldots, n``, the product ambiguity set ``\Gamma \subseteq \mathcal{D}(S)`` is defined as: 
```math
    \Gamma = \left\{ \gamma \in \mathcal{D}(S) \,:\, \gamma(s) = \prod_{i=1}^n \gamma^i(s^i), \, \gamma^i \in \Gamma_i \right\}
```
where ``s = (s_1, \ldots, s_n) \in S``. We will denote the product ambiguity set as ``\Gamma = \bigotimes_{i=1}^n \Gamma_i``. Each ``\Gamma_i`` is called a marginal or component ambiguity set.

## Factored RMDPs
Factored Robust Markov Decision Processes (fRMDPs) [schnitzer2025efficient, delgado2011efficient](@cite) are an extension of Robust Markov Decision Processes (RMDPs) [nilim2005robust, wiesemann2013robust, suilen2024robust](@cite) that incorporate a factored representation of the state and action spaces, i.e. with state and action variables. This allows for a more compact representation of the transition model and flexibility in modeling complex systems. First, we define here fRMDPs, and then in the subsequent sections, we define various special subclasses of fRMDPs, including how they relate to each other and to fRMDPs.

Formally, a fRMDP ``M`` is a tuple ``M = (S, S_0, A, \mathcal{G}, \Gamma)``, where

- ``S = S_1 \times \cdots \times S_n`` is a finite set of joint states with ``S_i`` being a finite set of states for the ``i``-th state variable,
- ``S_0 \subseteq S`` is a set of initial states,
- ``A = A_1 \times \cdots \times A_m`` is a finite set of joint actions with ``A_j`` being a finite set of actions for the ``j``-th action variable,
- ``\mathcal{G} = (\mathcal{V}, \mathcal{E})`` is a directed bipartite graph with nodes ``\mathcal{V} = \mathcal{V}_{ind} \cup \mathcal{V}_{cond} = \{S_1, \ldots, S_n, A_1, \ldots, A_m\} \cup \{S'_1, \ldots, S'_n\}`` representing the state and action variables and their next-state counterparts, and edges ``\mathcal{E} \subseteq \mathcal{V}_{ind} \times \mathcal{V}_{cond}`` representing dependencies of ``S'_i`` on ``S_j`` and ``A_k``,
- ``\Gamma = \{\Gamma_{s,a}\}_{s\in S,a \in A}`` is a set of ambiguity sets for source-action pair ``(s, a)``, where each ``\Gamma_{s,a} = \bigotimes_{i=1}^n \Gamma^i_{\text{Pa}_\mathcal{G}(S'_i) \cap (s, a)}`` is a product of ambiguity sets ``\Gamma^i_{\text{Pa}_\mathcal{G}(S'_i) \cap (s, a)}`` along each marginal ``i`` conditional on the values in ``(s, a)`` of the parent variables ``\text{Pa}_\mathcal{G}(S'_i)`` of ``S'_i`` in ``\mathcal{G}``, i.e.
```math
    \Gamma_{s,a} = \left\{ \gamma \in \mathcal{D}(S) \,:\, \gamma(s') = \prod_{i=1}^n \gamma^i(s'_i | s_{\text{Pa}_{\mathcal{G}_S}(S'_i)}, a_{\text{Pa}_{\mathcal{G}_A}(S'_i)}), \, \gamma^i(\cdot | s_{\text{Pa}_{\mathcal{G}_S}(S'_i)}, a_{\text{Pa}_{\mathcal{G}_A}(S'_i)}) \in \Gamma^i_{\text{Pa}_\mathcal{G}(S'_i)} \right\}.
```

A path of an fRMDP is a sequence of states and actions ``\omega = (s[0], a[0]), (s[1], a[1]), \dots`` where ``(s[k], a[k]) \in S \times A`` for all ``k \in \mathbb{N}_0``. We denote by ``\omega[k] = s[k]`` the state of the path at time ``k \in \mathbb{N}_0`` and by ``\Omega`` and ``\Omega_{fin}`` the set of all infinite and finite paths, respectively.
A _strategy_ or _policy_ for an fRMDP is a function ``\pi : \Omega_{fin} \to A`` that assigns an action, given a (finite) path called the history. _Time-dependent_ Markov strategies are functions from state and time step to an action, i.e. ``\pi : S \times \mathbb{N}_0 \to A``. This can equivalently be described as a sequence of functions indexed by time ``\mathbf{\pi} = (\pi[0], \pi[1], \ldots)``. If ``\pi`` does not depend on time and solely depends on the current state, it is called a _stationary_ strategy. Similar to a strategy, an adversary ``\eta`` is a function that assigns a feasible distribution to a given state. The focus of this package is on dynamic uncertainties where the choice of the adversary is resolved at every time step, called dynamic uncertainty, and where the adversary has access to both the current state and action, called ``(s, a)``-rectangularity. We refer to [suilen2024robust](@cite) for further details on the distinction between static and dynamic uncertainties, types of rectangularity, and their implications. Given a strategy and an adversary, an fRMDP collapses to a finite (factored) Markov chain.

## IMCs
Interval Markov Chains (IMCs) [delahaye2011decision](@cite) are a subclass of fRMDPs and a generalization of Markov Chains (MCs), where the transition probabilities are not known exactly, but they are constrained to be in some probability interval.
Formally, an IMC ``M`` is a tuple ``M = (S, S_0, \Gamma)``, where

- ``S`` is a finite set of states,
- ``S_0 \subseteq S`` is a set of initial states,
- ``\Gamma = \{\Gamma_{s}\}_{s\in S}`` is a set of ambiguity sets for source state ``s``, where each ``\Gamma_{s}`` is an interval ambiguity set over ``S``.

An IMC is equivalent to an fRMDP where there is only one state variable, no action variables, and the ambiguity sets are interval ambiguity sets. The dependency graph is just two nodes ``S`` and ``S'`` with a single edge from the former to the latter. Paths and adversaries are defined similarly to fRMDPs.

## IMDPs
Interval Markov Decision Processes (IMDPs) [givan2000bounded, lahijanian2015formal](@cite), also called bounded-parameter MDPs, are a subclass of fRMDPs and a generalization of MDPs, where the transition probabilities, given source state and action, are not known exactly, but they are constrained to be in some probability interval. IMDPs generalized IMCs by adding actions.
Formally, an IMDP ``M`` is a tuple ``M = (S, S_0, A, \Gamma)``, where

- ``S`` is a finite set of states,
- ``S_0 \subseteq S`` is a set of initial states,
- ``A`` is a finite set of actions,
- ```\Gamma = \{\Gamma_{s,a}\}_{s\in S,a \in A}`` is a set of ambiguity sets for source-action pair ``(s, a)``, where each ``\Gamma_{s,a}`` is an interval ambiguity set over ``S``.

An IMDP is equivalent to an fRMDP where there is only one state variable, one action variable, and the ambiguity sets are interval ambiguity sets. The dependency graph is three nodes ``S``, ``A``, and ``S'`` with two edges ``S \rightarrow S'`` and ``A \rightarrow S'``. Paths and adversaries are defined similarly to fRMDPs.

## odIMDPs
Orthogonally-decoupled IMDPs (odIMDPs) [mathiesen2025scalable](@cite) are a subclass of fRMDPs designed to be more memory-efficient than IMDPs. The states are structured into an orthogonal, or grid-based, decomposition and the transition probability ambiguity sets, for each source-action pair, as a product of interval ambiguity sets along each marginal. 

Formally, an odIMDP ``M`` with ``n`` marginals is a tuple ``M = (S, S_0, A, \Gamma)``, where

- ``S = S_1 \times \cdots \times S_n`` is a finite set of joint states with ``S_i`` being a finite set of states for the ``i``-th marginal,
- ``S_0 \subseteq S`` is a set of initial states,
- ``A`` is a finite set of actions,
- ``\Gamma = \{\Gamma_{s,a}\}_{s\in S,a \in A}`` is a set of ambiguity sets for source-action pair ``(s, a)``, where each ``\Gamma_{s,a} = \bigotimes_{i=1}^n \Gamma^i_{s,a}`` with ``\Gamma^i_{s,a}`` is an interval ambiguity set over the ``i``-th marginal, i.e. over ``S_i``.

An odIMDP is equivalent to an fRMDP where the dependency graph is ``\mathcal{G} = (\mathcal{V}, \mathcal{E})`` with ``\mathcal{V} = \{S_1, \ldots, S_n, A\} \cup \{S'_1, \ldots, S'_n\}`` and ``\mathcal{E} = \{(S_i, S'_j) : i, j = 1, \ldots, n\} \cup \{(A_i, S'_j) : j = 1, \ldots, m, i = 1, \ldots, n\}``. In other words, each next-state variable ``S'_i`` depends on all state and action variables and the dependency graph is a complete bipartite graph. Paths, strategies, and adversaries are defined similarly to fRMDPs.

## fIMDPs
Factored IMDPs (fIMDPs) are a subclass of fRMDPs where each marginal ambiguity set is an interval ambiguity set, but where the dependency graph can be arbitrary. 
Formally, an fIMDP ``M`` with ``n`` marginals is a tuple ``M = (S, S_0, A, \mathcal{G}, \Gamma)``, where

- ``S = S_1 \times \cdots \times S_n`` is a finite set of joint states with ``S_i`` being a finite set of states for the ``i``-th marginal,
- ``S_0 \subseteq S`` is a set of initial states,
- ``A`` is a finite set of actions,
- ``\mathcal{G} = (\mathcal{V}, \mathcal{E})`` is a directed bipartite graph with nodes ``\mathcal{V} = \{S_1, \ldots, S_n, A_1, \ldots, A_m\} \cup \{S'_1, \ldots, S'_n\}`` representing the state and action variables and their next-state counterparts, and edges ``\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}`` representing dependencies of ``S'_i`` on ``S_j`` and ``A_k``,
- ``\Gamma = \{\Gamma_{s,a}\}_{s\in S,a \in A}`` is a set of ambiguity sets for source-action pair ``(s, a)``, where each ``\Gamma_{s,a} = \bigotimes_{i=1}^n \Gamma^i_{\text{Pa}_\mathcal{G}(S'_i) \cap (s, a)}`` with ``\Gamma^i_{\text{Pa}_\mathcal{G}(S'_i) \cap (s, a)}`` is an interval ambiguity set over the ``i``-th marginal, i.e. over ``S_i``, conditional on the values in ``(s, a)`` of the parent variables ``\text{Pa}_\mathcal{G}(S'_i)`` of ``S'_i`` in ``\mathcal{G}``.

## References
```@bibliography
Pages = ["models.md"]
Canonical = false
```