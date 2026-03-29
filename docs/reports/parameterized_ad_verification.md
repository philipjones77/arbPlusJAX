Last updated: 2026-03-27T00:00:00Z

# Parameterized AD Verification

This report is the audited proof ledger for production-facing parameterized families. Each row is an explicit runtime audit target whose two-direction AD contract is enforced by [test_parameterized_public_ad_audit.py](/tests/test_parameterized_public_ad_audit.py).

Policy references:
- [special_function_ad_standard.md](/docs/standards/special_function_ad_standard.md)
- [operational_jax_standard.md](/docs/standards/operational_jax_standard.md)
- [implicit_adjoint_operator_solve_standard.md](/docs/standards/implicit_adjoint_operator_solve_standard.md)

Audited parameterized cases: `22`
Verified in both directions: `22`

Interpretation:
- `argument_status=audited_by_test` means the main evaluation-variable gradient is executed in the owning audit test.
- `parameter_status=audited_by_test` means the continuous family/control-parameter gradient is executed in the owning audit test.
- Discrete selector/index arguments are intentionally excluded from this audit.

| surface | family | kind | argument direction | parameter direction | argument_status | parameter_status | verification_status |
|---|---|---|---|---|---|---|---|
| `arb_pow` | `core` | `public point scalar` | `x` | `y` | `audited_by_test` | `audited_by_test` | `verified` |
| `acb_hurwitz_zeta` | `core` | `public point scalar` | `s` | `a` | `audited_by_test` | `audited_by_test` | `verified` |
| `arb_bessel_j` | `bessel` | `public point scalar` | `z` | `nu` | `audited_by_test` | `audited_by_test` | `verified` |
| `arb_bessel_i` | `bessel` | `public point scalar` | `z` | `nu` | `audited_by_test` | `audited_by_test` | `verified` |
| `incomplete_bessel_k` | `bessel` | `public point service` | `z` | `nu` | `audited_by_test` | `audited_by_test` | `verified` |
| `incomplete_gamma_upper` | `gamma` | `public point service` | `z` | `s` | `audited_by_test` | `audited_by_test` | `verified` |
| `incomplete_gamma_lower` | `gamma` | `public point service` | `z` | `s` | `audited_by_test` | `audited_by_test` | `verified` |
| `double_gamma.ifj_barnesdoublegamma` | `barnes` | `public point service` | `z` | `tau` | `audited_by_test` | `audited_by_test` | `verified` |
| `hypgeom.arb_hypgeom_0f1` | `hypergeometric` | `public point scalar` | `z` | `b` | `audited_by_test` | `audited_by_test` | `verified` |
| `hypgeom.arb_hypgeom_1f1` | `hypergeometric` | `public point scalar` | `z` | `a` | `audited_by_test` | `audited_by_test` | `verified` |
| `hypgeom.arb_hypgeom_2f1` | `hypergeometric` | `public point scalar` | `z` | `a` | `audited_by_test` | `audited_by_test` | `verified` |
| `hypgeom.arb_hypgeom_u` | `hypergeometric` | `public point scalar` | `z` | `a` | `audited_by_test` | `audited_by_test` | `verified` |
| `hypgeom.arb_hypgeom_pfq` | `hypergeometric` | `public point scalar` | `z` | `a[0]` | `audited_by_test` | `audited_by_test` | `verified` |
| `hypgeom.acb_hypgeom_0f1` | `hypergeometric` | `public point scalar` | `z` | `b` | `audited_by_test` | `audited_by_test` | `verified` |
| `hypgeom.acb_hypgeom_1f1` | `hypergeometric` | `public point scalar` | `z` | `a` | `audited_by_test` | `audited_by_test` | `verified` |
| `hypgeom.acb_hypgeom_2f1` | `hypergeometric` | `public point scalar` | `z` | `a` | `audited_by_test` | `audited_by_test` | `verified` |
| `hypgeom.acb_hypgeom_u` | `hypergeometric` | `public point scalar` | `z` | `a` | `audited_by_test` | `audited_by_test` | `verified` |
| `hypgeom.acb_hypgeom_pfq` | `hypergeometric` | `public point scalar` | `z` | `a[0]` | `audited_by_test` | `audited_by_test` | `verified` |
| `jrb_mat_operator_plan_apply` | `matrix` | `dense operator helper` | `v` | `scale` | `audited_by_test` | `audited_by_test` | `verified` |
| `srb_mat_matvec` | `matrix` | `sparse operator helper` | `v` | `scale` | `audited_by_test` | `audited_by_test` | `verified` |
| `jrb_mat_multi_shift_solve_point` | `matrix` | `matrix-free operator helper` | `rhs` | `shift` | `audited_by_test` | `audited_by_test` | `verified` |
| `curvature.make_posterior_precision_operator` | `matrix` | `curvature helper` | `v` | `damping` | `audited_by_test` | `audited_by_test` | `verified` |
