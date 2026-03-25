Last updated: 2026-03-25T00:00:00Z

# Dense Matrix Engineering Status

Scope:

- [arb_mat_implementation.md](/docs/implementation/modules/arb_mat_implementation.md)
- [acb_mat_implementation.md](/docs/implementation/modules/acb_mat_implementation.md)
- [dense_matrix_tranche_implementation.md](/docs/implementation/dense_matrix_tranche_implementation.md)

## Status Table

| family | point | basic | adaptive | rigorous | tightening | kernel_split | pure_jax | batch_recompile | ad | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| `arb_mat` dense real intervals | yes | yes | yes | yes | midpoint kernels plus perturbation-bounded determinant and residual-enclosed solve/inverse rigor | shared dispatch with separate point/basic/tighter kernels | mostly | fixed-shape cached plans and padded batch helpers exist | targeted chassis coverage | Large-`n` determinant rigor now uses midpoint-perturbation/Hadamard enclosure instead of aliasing `basic`; rigorous SPD/LU solve-family paths now inflate midpoint answers by residual bounds when the midpoint inverse is stable. |
| `acb_mat` dense complex boxes | yes | yes | yes | yes | midpoint kernels plus perturbation-bounded determinant and residual-enclosed solve/inverse rigor | shared dispatch with separate point/basic/tighter kernels | mostly | fixed-shape cached plans and padded batch helpers exist | targeted chassis coverage | Complex dense rigor now mirrors the real dense contract: large-`n` determinant uses complex perturbation/Hadamard boxes, and HPD/LU solve-family rigor uses residual-based enclosure inflation rather than pure midpoint boxing. |

## Remaining Dense Gaps

- deeper factorization-native rigorous semantics for `lu` / `qr` outputs themselves rather than only solve/inverse paths
- tighter large-`n` determinant bounds for badly conditioned midpoint matrices where the current Hadamard fallback is conservative
- broader dense reporting automation so this table can be generated from the capability registry instead of maintained manually
