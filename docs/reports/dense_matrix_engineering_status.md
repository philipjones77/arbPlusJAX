Last updated: 2026-03-25T00:00:00Z

# Dense Matrix Engineering Status

Scope:

- [arb_mat_implementation.md](/docs/implementation/modules/arb_mat_implementation.md)
- [acb_mat_implementation.md](/docs/implementation/modules/acb_mat_implementation.md)
- [dense_matrix_tranche_implementation.md](/docs/implementation/dense_matrix_tranche_implementation.md)

## Status Table

| family | point | basic | adaptive | rigorous | tightening | kernel_split | pure_jax | batch_recompile | ad | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| `arb_mat` dense real intervals | yes | yes | yes | yes | midpoint kernels plus perturbation-bounded determinant, cofactor-Lipschitz tightening, and residual-enclosed solve/inverse rigor | shared dispatch with separate point/basic/tighter kernels | mostly | fixed-shape cached plans and padded batch helpers exist | targeted chassis coverage | Large-`n` determinant rigor now combines midpoint-perturbation, Hadamard, and cofactor-Lipschitz bounds instead of aliasing `basic`; rigorous SPD/LU solve-family paths inflate midpoint answers by residual bounds when the midpoint inverse is stable, and factor outputs preserve unit-diagonal/triangular structural zeros. |
| `acb_mat` dense complex boxes | yes | yes | yes | yes | midpoint kernels plus perturbation-bounded determinant, cofactor-Lipschitz tightening, and residual-enclosed solve/inverse rigor | shared dispatch with separate point/basic/tighter kernels | mostly | fixed-shape cached plans and padded batch helpers exist | targeted chassis coverage | Complex dense rigor now mirrors the real dense contract: large-`n` determinant uses radius-corrected perturbation/Hadamard/cofactor boxes, HPD/LU solve-family rigor uses residual-based enclosure inflation rather than pure midpoint boxing, and rigorous factors preserve structural zeros/unit diagonals up to outward-rounding noise. |

## Remaining Dense Gaps

- broader dense reporting automation so this table can be generated from the capability registry instead of maintained manually
