Last updated: 2026-03-17T00:00:00Z

# Function Gap Plan

## Purpose

This document ranks the major remaining Arb/FLINT function gaps for arbPlusJAX.

The goal is not breadth-first parity. The goal is selective implementation:

- first for IFJ and RF77-facing needs
- then for broadly useful Arb-style surfaces
- and only later, if ever, for low-value overload or maintenance-heavy surfaces

Raw gap inventories live in:

- [audit.md](/home/phili/projects/arbplusJAX/docs/status/audit.md)
- [missing_impls/](/home/phili/projects/arbplusJAX/docs/status/reports/missing_impls)

## Current baseline

- The curated core subset is largely implemented already; see [core_function_status.md](/home/phili/projects/arbplusJAX/docs/status/reports/core_function_status.md).
- The broader Arb/FLINT callable surface is still far from complete.
- The biggest remaining non-helper gaps are in:
  - broader scalar special functions
  - dense matrix parity
  - polynomial algebra parity
  - Dirichlet and L-function work
  - elliptic and modular work

## Tier 1: High-Value For IFJ And RF77

- Barnes-family hardening first.
  - Priority functions: double gamma normalization/convention finalization, IFJ-derived Barnes double gamma transplant, `acb_barnes_g`, and compatible log variants.
  - Reason: this is foundational for Mellin-Barnes and higher Barnes constructions.

- Gamma-adjacent continuation functions that directly support contour and residue workflows.
  - Priority functions: rising factorial variants, lower beta, selected zeta/Hurwitz/Lerch-related functions where they unblock actual IFJ kernels.
  - Source gap files:
    - [arb_core_missing.txt](/home/phili/projects/arbplusJAX/docs/status/reports/missing_impls/arb_core_missing.txt)
    - [acb_core_missing.txt](/home/phili/projects/arbplusJAX/docs/status/reports/missing_impls/acb_core_missing.txt)
    - [acb_dirichlet_missing.txt](/home/phili/projects/arbplusJAX/docs/status/reports/missing_impls/acb_dirichlet_missing.txt)

- Select complex special functions that commonly appear in analytic continuation or transform formulas.
  - Priority families: `Ei`, `Chi`, `Ci`, dilogarithm, Tricomi `U`, and selected `pfq`.
  - Reason: these are more likely to be pulled into IFJ than broad Dirichlet machinery or dense matrix algebra.

- Contour/integration-facing function surfaces.
  - Priority functions: selective `acb_calc_integrate` parity only where the functionality improves current contour or line-integration work.
  - Source gap files:
    - [acb_calc_missing.txt](/home/phili/projects/arbplusJAX/docs/status/reports/missing_impls/acb_calc_missing.txt)
    - [arb_calc_missing.txt](/home/phili/projects/arbplusJAX/docs/status/reports/missing_impls/arb_calc_missing.txt)

## Tier 2: Broad Arb Parity With Good General Value

- Dense matrix parity in `arb_mat` and `acb_mat`.
  - Priority families: canonical `add`, `sub`, `mul`, `det`, `inv`, `solve`, `lu`, `charpoly`, `exp`, `trace`.
  - Complex-only next layer: eigenvalue and enclosure routines.
  - Source gap files:
    - [arb_mat_missing.txt](/home/phili/projects/arbplusJAX/docs/status/reports/missing_impls/arb_mat_missing.txt)
    - [acb_mat_missing.txt](/home/phili/projects/arbplusJAX/docs/status/reports/missing_impls/acb_mat_missing.txt)

- Polynomial algebra parity in `arb_poly` and `acb_poly`.
  - Priority families: compose, derivative, divrem, evaluate, integral, interpolation, `mullow`, product-of-roots, Taylor shift, root finding.
  - Source gap files:
    - [arb_poly_missing.txt](/home/phili/projects/arbplusJAX/docs/status/reports/missing_impls/arb_poly_missing.txt)
    - [acb_poly_missing.txt](/home/phili/projects/arbplusJAX/docs/status/reports/missing_impls/acb_poly_missing.txt)

- Broader scalar special functions beyond the current curated core.
  - Priority families: `agm`, `atan2`, `lambertw`, zeta variants, Bernoulli/Bell/Euler combinatorial functions, Chebyshev families, cotangent/cosecant families.
  - Source gap files:
    - [arb_core_missing.txt](/home/phili/projects/arbplusJAX/docs/status/reports/missing_impls/arb_core_missing.txt)
    - [acb_core_missing.txt](/home/phili/projects/arbplusJAX/docs/status/reports/missing_impls/acb_core_missing.txt)

## Tier 3: Valuable, But Lower Priority For Current Direction

- Elliptic and modular stacks.
  - Priority families: Weierstrass `p`, sigma, zeta, Carlson forms, eta, theta, lambda, Eisenstein.
  - Source gap files:
    - [acb_elliptic_missing.txt](/home/phili/projects/arbplusJAX/docs/status/reports/missing_impls/acb_elliptic_missing.txt)
    - [acb_modular_missing.txt](/home/phili/projects/arbplusJAX/docs/status/reports/missing_impls/acb_modular_missing.txt)
  - Reason: mathematically important, but not currently the tightest blocker for the IFJ/RF77 workstream.

- Dirichlet and full L-function stack.
  - Priority families: Dirichlet characters, `L`, Hardy `Z`, zeta zero tooling, Gauss/Jacobi sums, Lerch/Hurwitz support.
  - Source gap files:
    - [dirichlet_missing.txt](/home/phili/projects/arbplusJAX/docs/status/reports/missing_impls/dirichlet_missing.txt)
    - [acb_dirichlet_missing.txt](/home/phili/projects/arbplusJAX/docs/status/reports/missing_impls/acb_dirichlet_missing.txt)
  - Reason: large and valuable, but too broad to pursue before Barnes/gamma/selected contour needs are stabilized.

## Tier 4: Not Worth Prioritizing

- Typed overload variants that do not add new mathematical capability.
  - Examples: `_ui`, `_si`, `_fmpz`, `_fmpq` surface multiplication when the base function is still missing or not hardened.

- Low-level lifecycle, pointer, or printing surfaces.
  - Examples: `*_init`, `*_clear`, `*_allocated_bytes`, `*_entry_ptr`, `*_print*`, `*_fprint*`.

- Algorithm-choice internals and precompute/workspace helpers as first-class migration targets.
  - Examples: `_asymp`, `_direct`, `_bound`, `_choose`, `_precomp`, `_threaded`, `_classical`, `_recursive` variants.
  - These should usually follow a public-function decision, not lead it.

## Recommended near-term order

1. Finish Barnes-family hardening and make the IFJ-derived double-gamma path available side-by-side in arbPlusJAX.
2. Extend the stable special-function surface only where IFJ or RF77 actually consumes it.
3. Fill the highest-value scalar gaps next: `lambertw`, selected zeta/rising/beta/dilog/Ei-family functions.
4. Continue dense matrix and polynomial parity only where it improves direct downstream use rather than satisfying breadth for its own sake.
5. Leave full elliptic/modular and Dirichlet breadth until the Barnes/gamma/integration path is stable enough to justify a larger expansion.

## Rule

Do not migrate the missing Arb/FLINT surface breadth-first.

Prefer:

- one canonical implementation per important function family
- public JAX-native implementations over SciPy-derived runtime paths
- hardening and contracts on downstream-critical functions before widening the catalog
