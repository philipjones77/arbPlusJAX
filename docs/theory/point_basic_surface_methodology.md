Last updated: 2026-03-27T00:00:00Z

# Point And Basic Surface Methodology

## Purpose

This note records the repo-level mathematical and runtime interpretation of the
public `point` and `basic` surfaces.

It is the theory/methodology companion to:

- [point_surface_standard.md](/docs/standards/point_surface_standard.md)
- [point_fast_jax_standard.md](/docs/standards/point_fast_jax_standard.md)
- [point_basic_surface_status.md](/docs/reports/point_basic_surface_status.md)
- [parameterized_ad_verification.md](/docs/reports/parameterized_ad_verification.md)

## Surface Semantics

For a public family or function `f`:

- `S_point(f)` is the repeated-evaluation midpoint or helper-path surface used
  for fast JAX calling, stable-shape compilation, and service-style batch
  execution.
- `S_basic(f)` is the baseline enclosure/tightening surface. It is not required
  to be the strongest rigorous path, but it must preserve the family’s public
  containment interpretation.

In this repo, the two surfaces are related but not identical:

- `point` is the performance-first public contract.
- `basic` is the baseline containment contract.

The public API is therefore allowed to be:

- `point`-only for families whose public role is direct numeric evaluation or
  operator/matrix execution
- `point` plus `basic` for families that expose baseline enclosure behavior to
  callers

## Diagnostics Semantics

When a public family exposes diagnostics, the diagnostics payload `D_f` should
be interpreted as a public explanation surface for the chosen strategy rather
than as private debugging output.

Examples include:

- method selection
- regime tags
- convergence/residual metadata
- uncertainty estimates used to widen a `basic` output

The diagnostics payload should therefore line up with:

- public metadata
- benchmark summaries
- notebook explanations

## AD Direction Semantics

For parameterized families, the methodology must distinguish:

- argument-direction AD: differentiation through the primary evaluation
  variable
- parameter-direction AD: differentiation through a continuous family
  parameter

The repo does not treat these as interchangeable.

A family is only AD-complete in the current audit when both directions are
backed by:

- tests
- benchmarks
- canonical notebook sections

The repo now also keeps a practical runtime audit layer for the production
parameterized surfaces themselves:

- [parameterized_ad_verification.md](/docs/reports/parameterized_ad_verification.md)

That ledger is not a family summary. It is a checked list of actual public
surfaces and helper entrypoints whose argument-direction and
family-parameter-direction gradients are executed in pytest.

## Family-Level Interpretation

Current public behavior is not uniform across all families:

- core and special-function enclosure-oriented families often expose both
  `S_point(f)` and `S_basic(f)`
- matrix and matrix-free families are currently more point-first at the public
  metadata layer, while `basic` semantics often live in diagnostics-bearing
  helper surfaces and dedicated wrappers
- curvature is a cross-helper layer that consumes the matrix and matrix-free
  point/basic substrate rather than acting as a separate universal public mode
  family

This is why the family-level verification ledger is needed:

- [point_basic_surface_status.md](/docs/reports/point_basic_surface_status.md)

It records not just whether a family has point/basic exposure, but also whether
tests, benchmarks, notebooks, and diagnostics-bearing helper surfaces exist.

This is also why the runtime AD audit ledger is needed:

- [parameterized_ad_verification.md](/docs/reports/parameterized_ad_verification.md)

It records which production-facing parameterized surfaces are actually executed
through both AD directions, including matrix/operator helper layers and the
curvature posterior-precision helper surface.

## Numerical Regimes

The point/basic distinction matters most in the following regimes:

- containment-sensitive scalar and special-function evaluation
- prepared-plan matrix and matrix-free workloads where repeated point use is
  hot, but uncertainty needs to be surfaced in `basic`
- sparse/operator helper layers where diagnostics and uncertainty are often
  the only tractable baseline enclosure story

Theory notes for specific families should describe how their own regime logic
feeds into:

- `S_point(f)`
- `S_basic(f)`
- `D_f`

rather than treating those as unrelated runtime choices.
