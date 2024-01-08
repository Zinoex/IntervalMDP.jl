Interval Markov Decision Processes (IMDPs), also called bounded-parameter MDPs [1], are a generalization of MDPs, where the transition probabilities, given source state and action, are not known exactly, but they are constrained to be in some probability interval. 
Formally, an IMDP ``M`` is a tuple ``M = (S, S_0`, A, \overline{P}, \underline{P})``, where

- ``S`` is a finite set of states,
- ``S_0 \subseteq S`` is a set of initial states,
- ``A`` is a finite set of actions,
- ``\underline{P}: S \times A \times S  \to [0,1]`` is a function, where ``\underline{P}(s,a,s')`` defines the lower bound of the transition probability from state ``s\in S`` (source) to state ``s'\in S`` (destination) under action ``a \in A`,
- ``\overline{P}: S \times A \times S \to [0,1]`` is a function, where ``\xoverline{P}(s,a,s')`` defines the upper bound of the transition probability from state ``s\in S`` to state ``s'\in S`` under action ``a \in A`.

For each state-action pair ``(s,a) \in S \times A``, it holds that ``\sum_{s'\in S} \underline{P}(s,a,s') \leq 1 \leq \sum_{s'\in S} \overline{P}(s,a,s')`` and a transition probability distribution ``p_{s,a}:S\to[0,1]`` is called _feasible_ if ``\underline{P}(s,a,s')\leq p_{s,a}(s')\leq\xoverline{P}(s,a,s')`` for all destinations ``s'\in S``. The set of all feasible distributions for the state-action pair ``(s,a)`` is denoted by ``\Gamma_{s,a}``.

A path of an IMDP is a sequence of states and actions ``\omega = (s_0,a_0),(s_1,a_1),\dots``, where ``(s_i,a_i)\in S \times A``. We denote by ``\omega(k) = s_k`` the state of the path at time ``k \in \mathbb{N}^0`` and by ``\Omega`` the set of all paths.  
A _strategy_ or _policy_ for an IMDP is a function ``\pi`` that assigns an action to a given state of an IMDP. _Time-dependent_ strategies are function from state and time step to an action, i.e. ``\pi: S\times \mathbb{N}^0 \to A``. If ``\pi`` does not depend on time and solely depends on the current state, it is called a _stationary_ strategy. Similar to a strategy, an adversary ``\eta`` is a function that assigns a feasible distribution to a given state. Given a strategy and an adversary, an IMDP collapses to a finite Markov chain.

In this formal framework, we can describe computing reachability given a target set ``G`` and a horizon ``K \in \mathbb{N} \cup \{\infty\}`` as the following objective 

``\mathrm{opt}^{\pi}_{\pi}\mathrm{opt}^{\eta}_{\eta}\mathbb{P}_{\pi,\eta }\left[\omega \in \Omega \mid \exists k \in [0,K], \, \omega(k)\in G  \right],``

where ``\mathrm{opt}^{\pi},\mathrm{opt}^{\eta} \in \{\min, \max\}`` and ``\mathbb{P}_{\pi,\eta }`` is the probability of the Markov chain induced by strategy ``\pi`` and adversary ``\eta``.
When ``\mathrm{opt}^{\eta} = \min``, the solution is called optimal _pessimistic_ probability (or reward), and conversely is called optimal _optimistic_ probability (or reward) when ``\mathrm{opt}^{\eta} = \max``.
The choice of the min/max for the action and pessimistic/optimistic probability depends on the application. 

Discounted reward is fairly similar to reachability, but instead of a target set, we have a reward function ``r: S \to \mathbb{R}`` and a discount factor ``\gamma \in [0,1]``. The objective is then

``\mathrm{opt}^{\pi}_{\pi}\mathrm{opt}^{\eta}_{\eta}\mathbb{E}_{\pi,\eta }\left[\sum_{k=0}^{\infty} \gamma^k r(\omega(k)) \right].``

[1] Givan, Robert, Sonia Leach, and Thomas Dean. "Bounded-parameter Markov decision processes." Artificial Intelligence 122.1-2 (2000): 71-109.