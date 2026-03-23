Last updated: 2026-03-23T00:00:00Z

# Barnes And Double-Gamma Methodology

## Scope

This note documents the production methodology for the Barnes-family and
double-gamma surfaces in arbPlusJAX, including:

- `barnesg`
- `double_gamma`
- the `bdg_*` family
- IFJ-facing Barnes/double-gamma provider-oriented entrypoints

## Production Interpretation

This family matters because it is both numerically delicate and strategically
important for downstream contour, residue, and continuation workflows.

The production goal is therefore stronger than “function exists”:

- the family should be callable through stable public surfaces
- metadata and diagnostics should expose method and hardening level
- the family should be suitable for downstream provider-style integration

## Method Layers

The current Barnes/double-gamma stack should be interpreted as layered:

1. point/basic kernels for ordinary evaluation
2. rigorous/adaptive wrapper paths for enclosure-sensitive workloads
3. provider-oriented entrypoints for downstream use where the capability, not
   the internal module layout, is what matters

This is why the repo keeps both hardened numeric kernels and IFJ-style provider
bridges rather than treating the family as a one-off special case.

## Numerical Difficulties

The main hardening pressure points are:

- rapid growth/decay
- cancellation in logarithmic or continuation forms
- sensitivity near singular or residue-heavy regions
- higher runtime cost for rigorous/adaptive sampling-based bounds

For production use, this means method and regime choice must remain visible in
diagnostics and metadata.

## Provider Boundary Interpretation

Barnes/double-gamma is the first family that should become explicitly
provider-grade for downstream libraries.

That means:

- stable capability entrypoints
- public metadata sufficient for routing
- diagnostics strong enough to explain difficult-region behavior
- tests/benchmarks/examples that describe the real calling contract

The consumer library should adapt to these capabilities through a thin adapter
layer on its side rather than importing repo-internal helpers directly.

## Runtime And AD Interpretation

The production runtime story for this family follows the repo-wide model:

- keep dtype policy explicit
- keep compile-relevant routing controls explicit
- use cached/bound public surfaces where repeated calls matter
- keep diagnostics separate from the mandatory numeric hot path

AD claims for this family should remain explicit and honest in metadata rather
than inferred from mathematical smoothness alone.

## Relation To Benchmarks And Examples

For Barnes/double-gamma, benchmarks and examples are part of the hardening
story:

- benchmarks should show cold/warm/recompile behavior and difficult-region cost
- examples should show the real production calling pattern
- tests should cover both ordinary usage and the numerically fragile regimes

## Current Open Hardening Direction

The remaining methodology work is concentrated in:

- finishing hardening and characterization of `bdg_*`
- reducing rigorous/adaptive sampling cost
- strengthening provider-grade capability boundaries for downstream use
- improving benchmark and report coverage where diagnostics exist but packaging
  is still incomplete
