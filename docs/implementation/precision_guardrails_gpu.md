Last updated: 2026-03-16T00:00:00Z

# Precision Guardrails for GPU Krylov Loops

## Purpose

This note records the mixed-precision guardrails currently recommended for JAX-native matrix-free linear algebra in arbPlusJAX.

The basic policy is simple:

- do elementwise throughput work in the natural accelerator dtype,
- do reductions and scalar recurrence coefficients in `float64` or `complex128`,
- detect non-finite failures early,
- keep the estimator and adjoint logic numerically conservative.

This is particularly important for:

- Krylov recurrences,
- SLQ and stochastic trace estimation,
- log-determinant estimation,
- spectral quadrature,
- likelihood and objective scalars.

## 1. Why these guardrails matter

GPU-friendly dtypes such as `float16`, `bfloat16`, and `float32` are often entirely acceptable for:

- elementwise kernels,
- sparse matvecs,
- many local nonlinear transforms.

They are much less safe for:

- long reductions,
- dot products,
- orthogonality checks,
- log-sum-exp accumulators,
- spectral tail contributions,
- stochastic-trace estimator accumulation.

Those are exactly the places where matrix-function and random-field workflows become numerically fragile.

## 2. Operational policy

The working policy for this repository is:

1. elementwise math may remain in accelerator-friendly dtype;
2. all reductions should accumulate in `float64` / `complex128`;
3. Krylov scalar coefficients should be computed from high-precision inner products;
4. final scalar objectives should be formed in high precision;
5. non-finite outputs should trip a precision or fallback response rather than silently propagate.

In shorthand:

- fast elementwise path,
- careful scalar path.

## 3. Where fp64 accumulation should be enforced

The most important sites are:

- `sum`, `mean`, `var`, `std`,
- `dot` / `vdot`,
- norm calculations,
- `logsumexp`,
- quadrature panel sums,
- Hutchinson / Hutch++ / SLQ estimator aggregation,
- Lanczos / Arnoldi / CG recurrence scalars,
- Gram and orthogonality checks,
- final objective scalars in losses and evidence terms.

If only one rule is remembered, it should be:

- matvecs can stay cheap;
- reductions cannot.

## 4. Current repository implementation

To make that policy concrete, the repo now contains a small utility layer:

- [jax_precision.py](/src/arbplusjax/jax_precision.py)

Current helpers include:

- `safe_sum`
- `safe_mean`
- `safe_dot`
- `safe_vdot_real`
- `safe_norm`
- `kahan_sum`
- `safe_logsumexp`
- `all_finite`

These utilities promote reduction paths to `float64` or `complex128` even when the incoming arrays are lower precision.

## 5. Current path improvements

This policy is already used in a few immediate places:

- [elementary.py](/src/arbplusjax/elementary.py) now routes `logsumexp` through `safe_logsumexp`.
- [iterative_solvers.py](/src/arbplusjax/iterative_solvers.py) now uses safe high-precision norms and inner products in the CG path and related scalar checks.
- [krylov_solvers.py](/src/arbplusjax/krylov_solvers.py) now uses safe reduction helpers for norms and CG scalar recurrences.

This is not a full mixed-precision runtime policy yet, but it establishes the core rule in the places where silent reduction error is most dangerous.

## 6. Compensated summation

When `float64` accumulation is too expensive or unavailable, compensated summation is the next useful defense.

The repository utility layer includes:

- `kahan_sum`

That is particularly useful for:

- long spectral tails,
- near-cancellation,
- wide dynamic range in quadrature terms,
- stochastic estimator accumulation where the probe contributions vary strongly in magnitude.

## 7. Non-finite guards

The practical safety pattern is:

- perform risky kernels,
- check `isfinite`,
- escalate precision or widen bounds on failure.

The repo already does this widely for interval/box outputs. The current addition is simply to make the scalar reduction side more systematic.

## 8. What is not yet implemented

The following are still policy targets rather than first-class infrastructure:

- dynamic loss scaling for AD,
- automatic anomaly-triggered precision promotion,
- orthogonality-drift-triggered reorthogonalisation policies in the public matrix-free APIs,
- precision escalation based on probe variance or tail-mass diagnostics.

Those are reasonable next steps, but they should be added only where the triggering logic is explicit and testable.

## 9. Practical recommendation for RandomFields77-style work

For current large-scale JAX workflows:

- keep sparse/dense matvecs in the natural accelerator dtype where needed,
- keep all scalar recurrence coefficients and estimator accumulators in `float64`,
- use `kahan_sum` where tail cancellation is visible,
- keep custom-VJP estimator logic separate from any future external low-precision backend calls,
- treat non-finite detection as a first-class tripwire rather than a debug-only convenience.

That policy is cheap, easy to audit, and usually gives most of the benefit of a more elaborate mixed-precision stack.
