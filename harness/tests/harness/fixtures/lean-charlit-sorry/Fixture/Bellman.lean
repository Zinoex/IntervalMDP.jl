/-! Adversarial fixture (D3): a `sorry` between two `'"'` char literals must be detected.
    A string-only stripper would treat `"' ... '"` as one string literal and hide it. -/
namespace Fixture

def bellman (r v : Nat) : Nat := r + v

def q : Char := '"'
theorem helper {r v w : Nat} (h : v ≤ w) : bellman r v ≤ bellman r w := by
  sorry
def q2 : Char := '"'
def q3 : Char := '\''
def q4 : Char := '\\'

theorem bellman_monotone {r v w : Nat} (h : v ≤ w) : bellman r v ≤ bellman r w := helper h

end Fixture
