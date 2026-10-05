/-! Adversarial fixture (D5): kernel-bypass / trusted-code constructs that need
    explicit proof-policy approval. -/
namespace Fixture

def bellman (r v : Nat) : Nat := r + v

def fastBellman (r v : Nat) : Nat := v + r

@[implemented_by fastBellman]
def bellmanImpl (r v : Nat) : Nat := r + v

@[extern "fixture_bellman"]
def bellmanExtern (r v : Nat) : Nat := r + v

unsafe def bellmanUnsafe (r v : Nat) : Nat := r + v

set_option debug.skipKernelTC true in
theorem bellman_monotone {r v w : Nat} (h : v ≤ w) : bellman r v ≤ bellman r w := by
  unfold bellman
  exact Nat.add_le_add_left h r

end Fixture
