Last updated: 2026-03-16T18:10:00Z

# Theory

This section collects mathematical background and derivation notes for the interval and special-function machinery used in arbPlusJAX.

It is not limited to pure derivations. The theory section also records the
mathematical interpretation of current production surfaces, especially where the
implementation distinguishes:

- true enclosure kernels versus midpoint-first approximations
- point/basic/adaptive/rigorous mode behavior
- direct versus cached/operator-plan execution strategies
- asymptotic, recurrence, quadrature, or transform-based special-function paths

## Coverage Map

The current theory notes cover:

- interval/box semantics and four-mode behavior
- scalar/core methodology
- matrix interval and operator methodology
- Bessel-family methodology
- selected sparse matrix-free logdet and selected-inversion directions

The current theory notes still need further expansion for:

- hypergeometric production methodology
- gamma / incomplete-gamma production methodology
- Barnes / double-gamma production methodology
- transform / NUFFT methodology
- sparse/block/vblock production methodology

Those are the main remaining theory-documentation gaps for the current CPU
production tranche.

## Production Interpretation

Current production usage should read the theory notes together with:

- [engineering_standard.md](/home/phili/projects/arbplusJAX/docs/standards/engineering_standard.md)
- [example_notebook_standard.md](/home/phili/projects/arbplusJAX/docs/standards/example_notebook_standard.md)

The theory documents explain what the algorithms and modes mean.
The standards explain how those algorithms should be exposed, benchmarked,
diagnosed, and taught through notebooks.

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
- [Gamma Family Methodology](/home/phili/projects/arbplusJAX/docs/theory/gamma_family_methodology.md)
  - Documents the incomplete gamma production stack and its complement, tail, diagnostics, and derivative interpretation.
- [Transform FFT NUFFT Methodology](/home/phili/projects/arbplusJAX/docs/theory/transform_fft_nufft_methodology.md)
  - Documents the DFT and NUFFT production interpretation, especially cached-plan reuse and direct versus accelerated execution strategy.
- [Sparse Symmetric Leja Plus Hutch++ Log-Det](/home/phili/projects/arbplusJAX/docs/theory/sparse_symmetric_leja_hutchpp_logdet.md)
  - Documents the sparse SPD `BCOO` log-determinant path in `jrb_mat`.
  - States the Newton-Leja action formula, Hutch++ trace estimator, adaptive stop rule, and sparse spectral-bound strategy.
- [Sparse Selected Inversion by Domain Decomposition](/home/phili/projects/arbplusJAX/docs/theory/sparse_selected_inversion_domain_decomposition.md)
  - Records the complementary sparse inverse-diagonal and selected-inverse direction suggested by `parsinv`.
  - States the overlap-domain decomposition idea, stochastic correction role, and the proposed JAX-native translation.

## Current Status

Status: `in_progress`

Why:

- the main interval/core/matrix/Bessel foundations are documented
- the theory index now reflects the current production-readiness categories
- several important production families still need dedicated methodology notes

The immediate next theory additions should be:

1. hypergeometric methodology
2. Barnes / double-gamma methodology
3. sparse/block/vblock production methodology
4. matrix-free production methodology beyond the existing logdet-specific note
