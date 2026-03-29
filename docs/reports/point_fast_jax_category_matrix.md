Last updated: 2026-03-23T00:00:00Z

# Point Fast JAX Category Matrix

This report states the required `point fast JAX` coverage across the six top-level repo functionality categories.

Governing documents:
- [fast_jax_standard.md](/docs/standards/fast_jax_standard.md)
- [point_fast_jax_implementation.md](/docs/implementation/point_fast_jax_implementation.md)
- [point_fast_jax_plan.md](/docs/status/point_fast_jax_plan.md)

| category | current tranche status | fast-point target | proof test surface |
|---|---|---|---|
| `1. core numeric scalars` | `representative proof landed` | direct JAX formulas and scalar kernels | scalar chassis and scalar API/service tests plus `test_point_fast_jax_categories.py` |
| `2. interval / box / precision modes` | `representative proof landed` | point wrappers, mode routing, and shape-stable dtype/precision handling | wrapper, mode, and precision-routing tests plus `test_point_fast_jax_categories.py` |
| `3. dense matrix functionality` | `representative proof landed` | point-mode dense apply, cached matvec/rmatvec, and structured helpers | dense chassis, plan, and structured tests plus `test_point_fast_jax_categories.py` |
| `4. sparse / block-sparse / vblock functionality` | `representative proof landed` | point-mode sparse apply and cached apply kernels | sparse chassis, format, and structured tests plus `test_point_fast_jax_categories.py` |
| `5. matrix-free / operator functionality` | `representative proof landed` | operator apply, cached operator actions, and point-safe estimators | matrix-free chassis, logdet, selected-inverse, and adjoint tests plus `test_point_fast_jax_categories.py` |
| `6. special functions` | `representative proof landed` | direct formulas, fixed recurrences, or approximant-backed point kernels | hypgeom, gamma, Bessel, incomplete-tail, and service tests plus `test_point_fast_jax_categories.py` |
