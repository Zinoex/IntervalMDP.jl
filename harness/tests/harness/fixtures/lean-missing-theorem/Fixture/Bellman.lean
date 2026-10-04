/-!
  Fixture: abstract (exact Nat arithmetic) Bellman-style update.
  Scope: mathematical model only -- says nothing about Julia Float64/GPU code.
  The words sorry and admit inside comments must NOT be flagged.
-/
namespace Fixture

/-- `bellman r v = r + v` (abstract one-step update). -/
def bellman (r v : Nat) : Nat := r + v

-- monotonicity of the update (no sorry here)
theorem bellman_mono_renamed {r v w : Nat} (h : v ≤ w) : bellman r v ≤ bellman r w := by
  unfold bellman
  exact Nat.add_le_add_left h r

end Fixture
