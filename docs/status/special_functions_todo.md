Last updated: 2026-03-28T00:00:00Z

# Special Functions TODO

This file tracks the special-function hardening backlog separately from the
general repo TODO.

Status legend:
- `done`: landed in code and covered at least by targeted tests
- `in_progress`: partially implemented or exposed, but still needs hardening
- `planned`: accepted roadmap item, not yet at implementation level

## Current Status

Status: `in_progress`

- `done`
  - canonical example notebooks now exist for top-level gamma and
    Barnes/double-gamma production surfaces, with explicit production-calling
    guidance and benchmark-extension notes
  - dedicated theory notes now exist for the gamma-family production stack
  - dedicated theory notes now exist for hypergeometric and Barnes/double-gamma
    production methodology
- `in_progress`
  - normalize special-function service benchmarks and diagnostics reporting more
    fully across hypergeometric, Bessel, gamma, and Barnes families
  - direct normalized special benchmark coverage now also includes
    `benchmark_hypgeom_extra.py`
  - continue converting notebook and benchmark guidance into schema-backed
    artifacts rather than stdout-only summaries

## Hardening Backlog

Status: `in_progress`

- continue hardening ordinary gamma, Barnes-family, and ordinary Bessel
  families where coverage remains uneven
- continue calibrating the generic tail-engine recurrence and sequence logic
  across more families
- bring incomplete `I` to the same regime maturity as incomplete `K`
- finish hardening and characterization of the `bdg_*` Barnes and
  double-gamma family
- keep extending Barnes/double-gamma diagnostics and provider contracts beyond
  the current scalar IFJ surface
- reduce runtime cost of rigorous/adaptive `bdg_*` samplers in
  [ball_wrappers.py](/src/arbplusjax/ball_wrappers.py)
- continue hypergeometric engineering cleanup:
  helper consolidation, family-specific adaptive/rigorous kernels, and
  compile-noise reduction outside the current representative families
- `pfq` fixed/padded basic and adaptive/rigorous mode-batch proofs are now
  landed on the canonical real/complex paths
- alternative hypergeometric hardening is now stronger:
  Boost `pfq` fixed/padded mode-batch proofs and helper/`pfq` point-AD smoke
  are landed, and CuSF `hyp1f1`/`hyp2f1` now have explicit mode containment
  plus point-AD checks
- regularized Boost `0f1`/`1f1` fixed-vs-padded containment and reciprocal
  `pfq` fixed-vs-padded/mode-containment proofs are now landed
- Boost helper aliases now have explicit cross-mode consistency checks
- direct owner tests now exist for `bessel_kernels` and `barnesg`
- extend benchmark and RF77-facing usage/report coverage where diagnostics
  exist but packaging is still incomplete

## Planned Additions

Status: `planned`

- extend the general incomplete-tail engine to more hypergeometric-tail
  families
- add multivariate Bessel work only after scalar incomplete infrastructure is
  genuinely stable
- add incomplete multivariate-Bessel-type routines only if reductions justify
  them
- resolve what true arbitrary precision means under a strict pure-JAX
  constraint

## Priority Rule For Remaining Breadth

Status: `in_progress`

- do not migrate the missing callable surface breadth-first
- prefer one canonical JAX-native implementation per important function family
- prioritize IFJ and RF77-facing work first:
  - Barnes-family hardening and IFJ-derived double-gamma work
  - gamma-adjacent continuation functions that unblock contour and residue
    workflows
  - selected complex special functions with direct downstream use:
    `Ei`, `Chi`, `Ci`, dilogarithm, Tricomi `U`, and selected `pfq`
- after that, prioritize broad-value parity:
  - dense matrix parity in `arb_mat` / `acb_mat`
  - polynomial parity in `arb_poly` / `acb_poly`
  - selected scalar gaps such as `lambertw`, zeta-adjacent functions, and
    rising/beta families
- defer broad elliptic/modular and full Dirichlet/L-function expansion until
  the Barnes/gamma/integration path is stable enough to justify it
