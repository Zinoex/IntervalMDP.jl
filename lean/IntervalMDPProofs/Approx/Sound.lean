import IntervalMDPProofs.Models.Specification

/-!
# Sound approximations

A computed value `W` is **sound** for satisfaction mode `m` with respect to the exact value `V*`
when it errs only in the conservative direction, pointwise on states:

* `pessimistic`: `W ≤ V*` (a guaranteed lower bound);
* `optimistic`: `W ≥ V*` (a guaranteed upper bound).

(`W` is the spec's `Ṽ`; Lean identifiers cannot contain the combining tilde.)

Every soundness theorem of the project is stated with `Sound m` (spec § Approximation Soundness);
end-to-end statements go through `IntervalMDP.Approx.iter_sound` (`Approx/Lift.lean`).

Julia counterpart: none directly; this is the correctness notion for the approximations of
IntervalMDP.jl (early stopping in `src/robust_value_iteration.jl`, recursive O-max and McCormick in
`src/bellman.jl`, IVI in `src/interval_value_iteration.jl`).
-/

namespace IntervalMDP.Approx

open Filter Topology

variable {S : Type*}

/-- `Sound m W V` — the approximation `W` of the exact value `V` errs only conservatively for the
satisfaction mode `m`: `W ≤ V` if `m = pessimistic`, `V ≤ W` if `m = optimistic` (pointwise).

Julia counterpart: the `satisfaction` field (`Pessimistic`/`Optimistic`) of `Specification`
(`src/specification.jl`) decides which direction is conservative. -/
def Sound : SatisfactionMode → (S → ℝ) → (S → ℝ) → Prop
  | .pessimistic, W, V => W ≤ V
  | .optimistic, W, V => V ≤ W

/-- `StepSound m T' T` — the approximate one-step operator `T'` is sound for the exact operator
`T` at every input: `Sound m (T' V) (T V)` for all `V`.

Julia counterpart: one `bellman!` call (`src/bellman.jl`) against the exact Bellman operator;
the conservative direction is the `satisfaction` field of `Specification`
(`src/specification.jl`). -/
def StepSound (m : SatisfactionMode) (T' T : (S → ℝ) → (S → ℝ)) : Prop :=
  ∀ V, Sound m (T' V) (T V)

/-- In pessimistic mode, soundness is `W ≤ V`.

Julia counterpart: `Pessimistic` of `SatisfactionMode` (`src/specification.jl`). -/
@[simp] theorem sound_pessimistic {W V : S → ℝ} : Sound .pessimistic W V ↔ W ≤ V := Iff.rfl

/-- In optimistic mode, soundness is `V ≤ W`.

Julia counterpart: `Optimistic` of `SatisfactionMode` (`src/specification.jl`). -/
@[simp] theorem sound_optimistic {W V : S → ℝ} : Sound .optimistic W V ↔ V ≤ W := Iff.rfl

/-- The exact value is a sound approximation of itself.

Julia counterpart: none (Lean-side proof device). -/
theorem Sound.refl (m : SatisfactionMode) (V : S → ℝ) : Sound m V V := by
  cases m <;> exact (le_refl V : V ≤ V)

/-- Soundness is transitive: a sound approximation of a sound approximation is sound.

Julia counterpart: none (Lean-side proof device). -/
theorem Sound.trans {m : SatisfactionMode} {V₁ V₂ V₃ : S → ℝ} (h₁₂ : Sound m V₁ V₂)
    (h₂₃ : Sound m V₂ V₃) : Sound m V₁ V₃ := by
  cases m
  · exact le_trans (α := S → ℝ) h₁₂ h₂₃
  · exact le_trans (α := S → ℝ) h₂₃ h₁₂

/-- A monotone operator preserves soundness.

Julia counterpart: none (Lean-side proof device). -/
theorem Sound.apply_mono {m : SatisfactionMode} {T : (S → ℝ) → (S → ℝ)} (hT : Monotone T)
    {W V : S → ℝ} (h : Sound m W V) : Sound m (T W) (T V) := by
  cases m
  · exact hT h
  · exact hT h

/-- Soundness passes to limits: if `Wₙ` is sound for `Vₙ` for every `n`, `Wₙ → L'` and `Vₙ → L`,
then `L'` is sound for `L` (the pointwise order on `S → ℝ` is closed).

Julia counterpart: none (Lean-side proof device). -/
theorem Sound.of_tendsto {m : SatisfactionMode} {W V : ℕ → S → ℝ} {L' L : S → ℝ}
    (hW : Tendsto W atTop (𝓝 L')) (hV : Tendsto V atTop (𝓝 L)) (h : ∀ n, Sound m (W n) (V n)) :
    Sound m L' L := by
  cases m
  · exact le_of_tendsto_of_tendsto' hW hV h
  · exact le_of_tendsto_of_tendsto' hV hW h

end IntervalMDP.Approx
