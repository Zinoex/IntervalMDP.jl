import IntervalMDPProofs.Models.Distribution
import IntervalMDPProofs.Models.AmbiguitySet
import IntervalMDPProofs.Models.IntervalAmbiguity
import IntervalMDPProofs.Models.RMDP
import IntervalMDPProofs.Models.IMDP
import IntervalMDPProofs.Models.Factored
import IntervalMDPProofs.Models.DFA
import IntervalMDPProofs.Models.Product
import IntervalMDPProofs.Models.Strategy
import IntervalMDPProofs.Models.Specification
import IntervalMDPProofs.Models.Examples
import IntervalMDPProofs.Approx.Sound
import IntervalMDPProofs.Approx.Lift
import IntervalMDPProofs.Index.Julia
import IntervalMDPProofs.Index.Linear
import IntervalMDPProofs.Index.Sparse
import IntervalMDPProofs.Index.Perm
import IntervalMDPProofs.Index.Marginal
import IntervalMDPProofs.Index.Strategy
import IntervalMDPProofs.OMax
import IntervalMDPProofs.Bellman

/-!
# IntervalMDPProofs

Lean 4 models and proofs for the value-iteration / Bellman algorithms of IntervalMDP.jl.
See `README.md` for the model hierarchy, the Julia ↔ Lean glossary and the file map.
-/
