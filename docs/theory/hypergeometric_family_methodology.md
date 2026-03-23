Last updated: 2026-03-23T00:00:00Z

# Hypergeometric Family Methodology

## Scope

This note documents the production methodology for the hypergeometric-facing
surfaces in arbPlusJAX, including:

- ordinary gamma-adjacent functions exposed through `hypgeom`
- confluent and Gauss hypergeometric representatives such as `1F1`, `2F1`, and
  Tricomi `U`
- incomplete gamma upper/lower continuation surfaces
- Bessel-, Airy-, Fresnel-, and error-function-adjacent helper paths when they
  are routed through the shared hypergeometric stack

## Production Interpretation

The hardened public story for this family is not “one universal series code”.
It is a routed stack of midpoint kernels, rigorous/adaptive wrappers, and
family-specific special handling for difficult regimes.

In production terms, the important layers are:

1. point/basic midpoint kernels for the fast path
2. rigorous/adaptive wrappers when enclosure quality matters
3. metadata/diagnostics strong enough to tell downstream code which regime or
   method is active

## Method Families

The current hypergeometric stack uses several numerical ideas rather than one
global methodology:

- direct special-case kernels for gamma/error/Bessel-adjacent functions
- series-style evaluation for representative `pfq` families
- complementary-form selection for incomplete gamma upper/lower paths
- adaptive sampling and Jacobian/Lipschitz enclosure logic through the shared
  wrapper layer when exact analytic bounds are not the only path

This is why `hypgeom_wrappers` exists as a family router rather than a thin
alias layer.

## Regime Handling

The production difficulties in this family are dominated by:

- near-cancellation in complementary forms
- parameter regimes close to poles or singular recurrences
- branch-sensitive complex continuation
- compile noise from many function-specific variants if routing is not shared

The current hardened direction is therefore:

- keep family-specific kernels where they materially improve stability
- keep shared wrapper dispatch for mode/dtype/runtime policy
- keep regime distinctions explicit in metadata/diagnostics instead of hiding
  them behind a single undifferentiated interface

## Incomplete Gamma Rule

Incomplete gamma upper/lower is an important special case because direct and
complementary forms can have meaningfully different numerical behavior.

The wrapper methodology therefore prefers:

- compute the direct candidate
- compute the complementary candidate
- select the tighter resulting enclosure

That is a production hardening rule, not just an implementation detail.

## AD And JIT Interpretation

The production AD/JIT story for the hypergeometric stack is:

- keep value parameters dynamic where shapes remain stable
- keep compile-relevant method/routing controls explicit
- allow rigorous/adaptive paths to exist without forcing all hot calls through
  the most expensive enclosure logic

Examples and benchmarks should therefore teach:

- stable dtype selection
- binder reuse when routed through the public API
- optional padding/chunking where batch traffic would otherwise trigger
  recompilation churn

## Relation To Tests And Benchmarks

For this family, tests and benchmarks should be interpreted as part of the
methodology contract:

- tests own correctness, mode dispatch, and difficult-regime regressions
- benchmarks own cold/warm/recompile behavior and service-facing calling cost
- notebooks teach the public production pattern rather than only showing one
  mathematical identity

## Current Open Hardening Direction

The remaining methodology work is concentrated in:

- stronger benchmark/diagnostic normalization across the broader hypergeometric
  surface
- more explicit regime metadata
- further cleanup of family-specific rigorous/adaptive kernels
- continued reduction of compile-noise across representative `pfq` families
