import IntervalMDPProofs

/-!
# Documentation lint

Runs Mathlib/Batteries' `docBlame` (definitions, structures) and `docBlameThm` (theorems) environment
linters on the library. The build itself enables `linter.missingDocs` (see `lakefile.toml`), which
covers definitions only. Run from `lean/`:

    lake env lean DocLint.lean
-/

#lint only docBlame docBlameThm in IntervalMDPProofs
