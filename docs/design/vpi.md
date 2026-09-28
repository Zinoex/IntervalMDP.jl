# VPI successor score for trajectory sampling

A design note on `TrajectorySampling.VPIScore` (`src/trajectorysampling.jl`, §4b): the
value-of-perfect-information heuristic of VPI-RTDP (Sanner, Goetschalckx, Driessens & Shani,
*Bayesian Real-Time Dynamic Programming*, IJCAI 2009), restated for interval MDPs.

> **Scope.** The note covers only the successor score. No termination rule or
> priority-sweeping priority is built on VPI.

---

## 1. Motivation

The existing successor scores (`ExplorationScore`, `GapWeightedExplorationScore`) and the
BRTDP stopping rule (`ExpectedGapStop`) all steer by the value **gap** `U(t) − L(t)`. So they
spend backups on successors whose gap is open even when closing it could not change the
action chosen at the state the rollout came from. VPI asks a different question: *if an oracle
told us `V*(t)`, how much better could the decision at `s` become?*

---

## 2. Notation

At the rollout's current state `s`, with `a*` the action the rollout **just sampled**:

| Symbol | Meaning | In code |
|---|---|---|
| `U(t), L(t)` | value bounds at successor `t` | `vf.upper.current`, `vf.lower.current` |
| `U(s,a), L(s,a)` | O-max Q-bounds of `a` at `s` | `dot(pU, U)`, `dot(pL, L)` with `pU/pL = _omax_distribution(marginal[a,s], U/L, dir)` |
| `p_a(t)` | the distribution the strategy's `ConcreteTransition` realizes for `(s,a)` | `pU` or `pL` per `transition.bound`; for `a*` it is the rollout's `probs` |
| `σ` | `+1` for `Maximize`, `−1` for `Minimize` | `_ismaximize(spec)` |

Write `m(t) = (U(t)+L(t))/2` and `h(t) = (U(t)−L(t))/2`.

---

## 3. Derivation

**Belief.** As in the paper (§3.1), the only knowledge about `v_t` is its bracket, so
`v_t ~ Uniform[L(t), U(t)]`, independently across states.

**Expected Q-value (paper eq. 8).** With a fixed transition,
`E[Q_a] = R + Γ_a·(U+L)/2 = (Q^U_a + Q^L_a)/2`. In an IMDP the upper and lower Q-bounds are
realized by possibly different distributions, so we take the midpoint of the Q-interval:

```
m_a := (U(s,a) + L(s,a)) / 2
```

This is exact whenever the U- and L-realized distributions coincide (a precise MDP, or when both
O-max sort orders agree). Otherwise it amounts to a uniform belief over the action's own Q-bracket.

**Revealing `t` (eq. 9).** Replace `t`'s factor with a point mass at `v`. Only the `t` term of
the expectation moves. With `x = v − m(t) ∈ [−h(t), h(t)]`:

```
E[Q_a | v_t = v] = m_a + p_a(t) · x
```

This is one line per action, in the unknown `x`.

**Gain over `a*` (eq. 10).** Let `Δ_a = σ(m_a − m_{a*})` and `δ_a(t) = σ(p_a(t) − p_{a*}(t))`:

```
Gain_a(x) = max(0, Δ_a + δ_a(t)·x)
```

**VPI (eq. 11).** Average over the belief, then take the best alternative:

```
VPI_{s,a*}(t) = max_{a ≠ a*}  (1/2h) ∫_{−h}^{h} max(0, Δ_a + δ_a x) dx
```

### Closed form

The integration interval is symmetric about 0, so `x → −x` shows only `|δ_a|` matters. Define
the **swing**

```
k_a(t) = |p_a(t) − p_{a*}(t)| · h(t)
```

This is how far learning `v_t` can move the Q-difference. The average positive part of a
linear function over `[−h, h]` is then

```
φ(Δ, k) = 0               if Δ ≤ −k     (a never overtakes a*)
        = Δ               if Δ ≥  k     (a beats a* whatever v_t is)
        = (Δ + k)² / (4k) otherwise     (the triangle of the paper's Fig. 1)

VPI_{s,a*}(t) = max_{a ∈ A(s), a ≠ a*} φ(Δ_a, k_a(t))
```

In the mixed case the line is positive on a sub-interval of width `(Δ+k)/|δ|` and peaks at
`Δ+k`. The triangle's area is `(Δ+k)²/(2|δ|)`, and dividing by `2h` gives `(Δ+k)²/(4k)`.

Properties (all tested, `test/base/trajectorysampling.jl`, "VPI successor score"):

* `φ ≥ 0`, continuous and C¹ at `Δ = ±k`, and non-decreasing in both `Δ` and `k`.
* The third branch only runs when `k > |Δ|`, so there is no division by zero.
* `k = 0` (a converged `t`, or `p_a(t) = p_{a*}(t)`) gives `max(0, Δ)`. This is the paper's
  Dirac-delta case.
* Reward and discount: `U(s,a)` here is the raw `Σ p V`. A state reward cancels in `Δ`, and a
  discount `γ` scales `Δ` and `k`, and so VPI, by `γ`. Normalized selection is unchanged.

### Choice of `a*`

Following the paper's `CHOOSENEXTSTATE-VPI(s, a)`, `a*` is the action the rollout sampled, not
`argmax_a m_a`. The consequence is that when some `a` has a higher midpoint, `Δ_a > 0`, every
successor scores at least `Δ_a`, including converged ones. With `a* = argmax m_a` (Dearden et
al.'s proper VPI) we would have `Δ ≤ 0` throughout, and converged successors would score 0.

---

## 4. From VPI to a successor score

`VPIScore` follows the paper's Algorithm 3, minus its termination branch. With
`b(t) = p_{a*}(t)·(U(t) − L(t))`:

1. `max_t b(t) > β` → score = `b`. VPI tells you little while the bounds are nearly vacuous,
   so this falls back to BRTDP. The default `β = 0.95` assumes values in `[0, 1]`.
2. else if `Σ_t VPI(t) > 0` → score = `VPI`.
3. else → score = `b`. Whether the rollout ends is left to its `TerminationRule`s
   (e.g. `ExpectedGapStop`), not to the score.

Two deviations from the paper:

* **Candidates are `supp(p_{a*})`**, not every state. The paper's `t ~ v(·)/V` could otherwise
  move the rollout to a state that is not a successor of `(s, a*)`.
* **Proportional draw via `LogStateScore`.** The paper samples `t ∝ score`. In this vocabulary
  that is `Boltzmann(1.0)` over the log-score, as `LogScore` already does for actions:

```julia
TS.TrajectorySampling(;
    state_policy = TS.Boltzmann(1.0),
    state_score = TS.LogStateScore(TS.VPIScore()),
)
```

`EpsilonGreedy` keeps its documented asymmetry and exploits on `p·f_greedy` (the score's
`bound`), ignoring VPI.

---

## 5. Cost and plumbing

Each step computes two `_omax_distribution`s per available action (against `U` and `L`), then
`O(|A|·|supp|)` evaluations of `φ`. That is the same order as a Bellman backup at `s`, matching
the paper's complexity claim.

`_sample_state` now calls the nine-argument
`_state_scores(score, probs, supp, vf, s, a, model, spec, transition)`. Its generic
`StateScore` method forwards to the old four-argument form, so the existing scores are
unchanged.
