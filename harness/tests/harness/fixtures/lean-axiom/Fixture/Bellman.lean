/-!
  Fixture: abstract (exact Nat arithmetic) Bellman-style update.
  Scope: mathematical model only -- says nothing about Julia Float64/GPU code.
  The words sorry and admit inside comments must NOT be flagged.
-/
namespace Fixture

/-- `bellman r v = r + v` (abstract one-step update). -/
def bellman (r v : Nat) : Nat := r + v

axiom bellman_mono_ax : ∀ {r v w : Nat}, v ≤ w → bellman r v ≤ bellman r w

-- monotonicity of the update (no sorry here)
theorem bellman_monotone {r v w : Nat} (h : v ≤ w) : bellman r v ≤ bellman r w := by
  exact bellman_mono_ax h

end Fixture
