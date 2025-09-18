# Specifications

## Reachability
In this formal framework, we can describe computing reachability given a target set ``G`` and a horizon ``K \in \mathbb{N} \cup \{\infty\}`` as the following objective 

```math
{\mathop{opt}\limits_{\pi}}^{\pi} \; {\mathop{opt}\limits_{\eta}}^{\eta} \; \mathbb{P}_{\pi,\eta }\left[\omega \in \Omega : \exists k \in [0,K], \, \omega(k)\in G  \right],
```

where ``\mathop{opt}^{\pi},\mathop{opt}^{\eta} \in \{\min, \max\}`` and ``\mathbb{P}_{\pi,\eta }`` is the probability of the Markov chain induced by strategy ``\pi`` and adversary ``\eta``.
When ``\mathop{opt}^{\eta} = \min``, the solution is called optimal _pessimistic_ probability (or reward), and conversely is called optimal _optimistic_ probability (or reward) when ``\mathop{opt}^{\eta} = \max``.
The choice of the min/max for the action and pessimistic/optimistic probability depends on the application. 

## Discounted reward
Discounted reward is similar to reachability but instead of a target set, we have a reward function ``r: S \to \mathbb{R}`` and a discount factor ``\gamma \in (0, 1)``. The objective is then

```math
{\mathop{opt}\limits_{\pi}}^{\pi} \; {\mathop{opt}\limits_{\eta}}^{\eta} \; \mathbb{E}_{\pi,\eta }\left[\sum_{k=0}^{K} \gamma^k r(\omega(k)) \right].
```

[1] Givan, Robert, Sonia Leach, and Thomas Dean. "Bounded-parameter Markov decision processes." Artificial Intelligence 122.1-2 (2000): 71-109.

[2] Suilen, M., Badings, T., Bovy, E. M., Parker, D., & Jansen, N. (2024). Robust Markov Decision Processes: A Place Where AI and Formal Methods Meet. In Principles of Verification: Cycling the Probabilistic Landscape: Essays Dedicated to Joost-Pieter Katoen on the Occasion of His 60th Birthday, Part III (pp. 126-154). Cham: Springer Nature Switzerland.

[3] Mathiesen, F. B., Haesaert, S., & Laurenti, L. (2024). Scalable control synthesis for stochastic systems via structural IMDP abstractions. arXiv preprint arXiv:2411.11803.