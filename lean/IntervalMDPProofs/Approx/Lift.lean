import IntervalMDPProofs.Approx.Sound

/-!
# Lifting one-step soundness through value iteration (A1)

If an approximate operator `T'` is sound for the exact operator `T` at every input and one of the
two is monotone, then value iteration with `T'` stays sound for value iteration with `T`: at every
iterate, in the limit, and at fixed points.

Every per-algorithm soundness result of later phases (recursive O-max, McCormick, early stopping,
IVI) must be obtained by applying `iter_sound` (spec § Approximation Soundness, row A1) instead of
re-proving the induction.

Julia counterpart: `_value_iteration!` (`src/robust_value_iteration.jl`), which applies the
Bellman operator chosen by the algorithm (`bellman!`, `src/bellman.jl`) repeatedly.
-/

namespace IntervalMDP.Approx

open Filter Topology

variable {S : Type*}

/-- **A1.** If `T'` is sound for `T` at every input (`StepSound m T' T`), `T` or `T'` is monotone, and
the initial values are sound (`Sound m W₀ V₀`), then

1. every iterate is sound: `Sound m (T'^[n] W₀) (T^[n] V₀)` for all `n`, and
2. the limits are sound: if `T'^[n] W₀ → L'` and `T^[n] V₀ → L`, then `Sound m L' L`.

Julia counterpart: repeated `bellman!` calls in `_value_iteration!`
(`src/robust_value_iteration.jl`). -/
theorem iter_sound {m : SatisfactionMode} {T' T : (S → ℝ) → (S → ℝ)}
    (hstep : StepSound m T' T) (hmono : Monotone T ∨ Monotone T')
    {W₀ V₀ : S → ℝ} (h₀ : Sound m W₀ V₀) :
    (∀ n, Sound m (T'^[n] W₀) (T^[n] V₀)) ∧
      ∀ {L' L : S → ℝ}, Tendsto (fun n => T'^[n] W₀) atTop (𝓝 L') →
        Tendsto (fun n => T^[n] V₀) atTop (𝓝 L) → Sound m L' L := by
  have hiter : ∀ n, Sound m (T'^[n] W₀) (T^[n] V₀) := by
    intro n
    induction n with
    | zero => exact h₀
    | succ n ih =>
      rw [Function.iterate_succ_apply', Function.iterate_succ_apply']
      rcases hmono with hT | hT'
      · exact (hstep _).trans (ih.apply_mono hT)
      · exact (ih.apply_mono hT').trans (hstep _)
  exact ⟨hiter, fun hW hV => Sound.of_tendsto hW hV hiter⟩

/-- **A1, fixed points.** If `W` is a fixed point of the sound approximate operator `T'`, `W` is
sound for `V₀`, and the exact iterates `T^[n] V₀` converge to `V*` (e.g. the exact value), then
`W` is sound for `V*`. Derived from `iter_sound`.

Julia counterpart: the converged result of `_value_iteration!`
(`src/robust_value_iteration.jl`), a numerical fixed point of `bellman!` (`src/bellman.jl`). -/
theorem fixedPoint_sound {m : SatisfactionMode} {T' T : (S → ℝ) → (S → ℝ)}
    (hstep : StepSound m T' T) (hmono : Monotone T ∨ Monotone T')
    {W V₀ Vstar : S → ℝ} (hfix : T' W = W) (h₀ : Sound m W V₀)
    (hconv : Tendsto (fun n => T^[n] V₀) atTop (𝓝 Vstar)) : Sound m W Vstar := by
  have hconst : (fun n => T'^[n] W) = fun _ => W := funext fun n => Function.iterate_fixed hfix n
  have hlim : Tendsto (fun n => T'^[n] W) atTop (𝓝 W) := by rw [hconst]; exact tendsto_const_nhds
  exact (iter_sound hstep hmono h₀).2 hlim hconv

end IntervalMDP.Approx
