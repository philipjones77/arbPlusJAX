Last updated: 2026-03-16T18:10:00Z

# Theory

This section collects mathematical background and derivation notes for the interval and special-function machinery used in arbPlusJAX.

## Available notes

- [Ball Arithmetic and the Four-Mode Model](/home/phili/projects/arbplusJAX/docs/theory/ball_arithmetic_and_modes.md)
  - Defines ball arithmetic in the Arb / FLINT sense.
  - Explains the arbPlusJAX `point`, `basic`, `adaptive`, and `rigorous` modes.
  - Compares the project’s float64 box semantics to true arbitrary-precision ball arithmetic.
- [Matrix Interval Arithmetic and Matrix Modes](/home/phili/projects/arbplusJAX/docs/theory/matrix_interval_and_modes.md)
  - Defines the real-interval and complex-box matrix models used in arbPlusJAX.
  - Explains which matrix operations are direct enclosure computations and which are midpoint-first.
  - Compares the matrix `point`, `basic`, `adaptive`, and `rigorous` paths.
- [Elementary Functions Methodology](/home/phili/projects/arbplusJAX/docs/theory/elementary_functions_methodology.md)
  - Documents the mathematical helper layer in `elementary.py`.
  - States the formulas and role of each public elementary helper.
- [Core Functions Methodology](/home/phili/projects/arbplusJAX/docs/theory/core_functions_methodology.md)
  - Documents the real interval core in `arb_core.py` and the complex box core in `acb_core.py`.
  - States the base formulas, grouping, and mode inheritance for the core kernels.
- [Bessel Family Methodology](/home/phili/projects/arbplusJAX/docs/theory/bessel_family_methodology.md)
  - Documents the Hankel and spherical Bessel surfaces in `special/bessel`.
  - States the direct identities, recurrences, derivatives, and large-argument asymptotics used today.
- [Sparse Symmetric Leja Plus Hutch++ Log-Det](/home/phili/projects/arbplusJAX/docs/theory/sparse_symmetric_leja_hutchpp_logdet.md)
  - Documents the sparse SPD `BCOO` log-determinant path in `jrb_mat`.
  - States the Newton-Leja action formula, Hutch++ trace estimator, adaptive stop rule, and sparse spectral-bound strategy.
- [Sparse Selected Inversion by Domain Decomposition](/home/phili/projects/arbplusJAX/docs/theory/sparse_selected_inversion_domain_decomposition.md)
  - Records the complementary sparse inverse-diagonal and selected-inverse direction suggested by `parsinv`.
  - States the overlap-domain decomposition idea, stochastic correction role, and the proposed JAX-native translation.
