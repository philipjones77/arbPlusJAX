Last updated: 2026-03-01T00:00:00Z

# validation

Validation utilities for sanity checks and test helpers.

## Notes

Used by tests to ensure bounds, shapes, and numeric consistency across modes.

## Relation to external references

Validation compares arbPlusJAX against:
- **C Arb (flint)**: interval containment and midpoint parity.
- **SciPy / JAX NumPy**: point‑value parity only.
- **mpmath / Mathematica**: high‑precision point comparisons (no interval semantics).
- **CubesselK (pure JAX backend)**: Bessel-K backend with point/basic/rigorous/adaptive modes and no external CUDA/shared-library dependency.

## Recent stress tests

- loggamma: complex near the negative real axis; JAX basic matches C midpoint tightly, JAX point tracks mpmath.
- bessel-k backend characterization run (pure JAX CubesselK): `results/benchmarks/cubesselk_compare_purejax_20260301/samples_256_seed_7/summary.csv`.
