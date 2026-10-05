/-! Adversarial fixture (D4): `axiom propext` inside `namespace Fixture` declares
    `Fixture.propext` -- an arbitrary new axiom, NOT the approved core `propext`. -/
namespace Fixture

def bellman (r v : Nat) : Nat := r + v

axiom propext : ∀ {r v w : Nat}, v ≤ w → bellman r v ≤ bellman r w

theorem bellman_monotone {r v w : Nat} (h : v ≤ w) : bellman r v ≤ bellman r w := propext h

end Fixture
