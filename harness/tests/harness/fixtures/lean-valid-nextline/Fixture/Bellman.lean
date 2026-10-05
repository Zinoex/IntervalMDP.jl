/-! Valid fixture: declaration name on the line after `theorem`, multi-line
    statement, and char/string literals that look like comment/string openers. -/
namespace Fixture

def bellman (r v : Nat) : Nat := r + v

def quote : Char := '"'
def dash : String := "-- not a comment /- nor this"

theorem
    bellman_monotone {r v w : Nat}
    (h : v ≤ w) :
    bellman r v ≤ bellman r w := by
  unfold bellman
  exact Nat.add_le_add_left h r

end Fixture
