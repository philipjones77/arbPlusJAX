Last updated: 2026-03-23T00:00:00Z

# Gamma Family Methodology

## Purpose

This note records the current mathematical interpretation of the gamma-family
surfaces used in arbPlusJAX, with emphasis on the production-facing incomplete
gamma stack.

It is not a proof document for every implementation detail.
It explains which identities and methodological choices the current public
surfaces rely on.

## Scope

This note covers:

- gamma and log-gamma style scalar surfaces
- incomplete gamma lower / upper
- regularized and non-regularized forms
- complement-based construction of the lower function
- tail-acceleration interpretation for the upper function
- current derivative and diagnostics interpretation

It does not yet cover:

- hypergeometric gamma-adjacent reductions in full detail
- Barnes / double-gamma methodology

## Primary Mathematical Identities

For `Re(s) > 0` and `z >= 0`, the classical upper and lower incomplete gamma
functions are

`Gamma(s, z) = integral_z^inf t^(s-1) exp(-t) dt`

and

`gamma(s, z) = integral_0^z t^(s-1) exp(-t) dt`.

They satisfy the complement identity

`gamma(s, z) + Gamma(s, z) = Gamma(s)`.

The regularized forms are

`P(s, z) = gamma(s, z) / Gamma(s)`

and

`Q(s, z) = Gamma(s, z) / Gamma(s)`.

## Production Interpretation In arbPlusJAX

### Upper incomplete gamma

The current production-facing `incomplete_gamma_upper` surface is interpreted
as the primary tail-specialized kernel.

The implementation is organized around the tail integral directly rather than
treating the upper function as a small wrapper over an unrelated special
function path.

The practical meaning is:

- the upper function owns the tail diagnostics
- the upper function owns the tail remainder estimate
- the upper function owns the main method-selection logic for fragile regimes

### Lower incomplete gamma

The current `incomplete_gamma_lower` surface is interpreted as a complement
construction on top of the upper specialization.

That means the current production methodology is:

1. compute the upper-tail value or interval
2. combine it with the complete gamma quantity
3. expose the lower function as a stable public surface

This is mathematically natural because the lower function is often less stable
to compute directly in regimes where the complement is numerically better
behaved.

## Method Selection

The current public metadata and diagnostics expose a method-oriented view of the
gamma stack.

The main methods are interpreted as:

- `quadrature`
  - direct numerical evaluation of the defining or transformed integral
- `aitken`
  - sequence acceleration for slowly convergent tail panels
- `wynn`
  - Wynn epsilon-style acceleration for difficult tail convergence
- `high_precision_refine`
  - fallback refinement path for fragile regimes
- `auto`
  - method-selection policy over the above paths

The point of this layering is not to claim exact Arb-style arbitrary-precision
ball arithmetic.
It is to provide a production-facing interpretation of:

- which numerical path was used
- whether fallback/refinement was needed
- how much tail remainder was estimated

## Modes

The public gamma-family interval interpretation follows the repo-wide four-mode
model:

- `point`
  - midpoint-style direct value path
- `basic`
  - outward interval/box wrapping around the current numerical estimate plus
    method-aware remainder information
- `adaptive`
  - tighter interval attempt with more work where justified
- `rigorous`
  - most conservative current interval wrapper, still bounded by present
    float64/JAX limitations

For the gamma family, these interval modes should be read as current enclosure
contracts, not as a claim of full arbitrary-precision Arb equivalence.

## Diagnostics Interpretation

The current diagnostics object for incomplete gamma is part of the production
contract.

Its fields should be interpreted as:

- `method`
  - the selected numerical path
- `chunk_count`, `panel_count`, `recurrence_steps`
  - rough work/structure indicators
- `estimated_tail_remainder`
  - the current remainder estimate used by interval expansion logic
- `instability_flags`
  - regime warnings, not guaranteed fatal errors
- `fallback_used`
  - whether the preferred path escalated to a safer refinement
- `precision_warning`
  - whether the result is considered fragile under the present precision policy

These diagnostics are important because the current CPU tranche treats
inspectable numerical behavior as part of production quality.

## Derivatives

The current incomplete-gamma stack exposes explicit derivative surfaces and a
custom-JVP-backed interpretation.

The intended meaning is:

- argument and parameter derivatives are part of the public engineering
  contract
- autodiff is not left entirely to opaque tracing through the quadrature code
- derivative support should remain consistent with the chosen complement and
  tail-specialization structure

## Relationship To Benchmarks And Notebooks

The canonical benchmark and notebook surfaces should be read with this theory in
mind:

- benchmark results are not only speed numbers; they also expose which method
  and parameterization style the surface expects
- example notebooks should demonstrate bound service calls, stable dtype and
  padding choices, and optional diagnostics inspection

## Current Limitations

This note describes the current production interpretation, not an ideal future
state.

The main current limitations are:

- interval rigor remains float64/JAX bounded
- GPU validation is not yet the active tranche
- some method-selection heuristics are still engineering-level rather than
  theorem-backed guarantees

That is acceptable as long as the public metadata, diagnostics, examples, and
status reports describe the current reality honestly.
