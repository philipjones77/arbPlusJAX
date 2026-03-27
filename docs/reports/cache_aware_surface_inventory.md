Last updated: 2026-03-24T00:00:00Z

# Cache-Aware Surface Inventory

This generated report records the current cache-aware public surfaces and the canonical examples and benchmarks that should demonstrate compliant reuse patterns.

Refresh with `python tools/cache_aware_surface_report.py` or the umbrella `python tools/check_generated_reports.py` path.

## Bound API Reuse Surfaces

| surface | purpose |
|---|---|
| `api.bind_point_batch` | bind reusable point-batch callable |
| `api.bind_point_batch_jit` | bind reusable compiled point-batch callable |
| `api.bind_interval_batch` | bind reusable interval-batch callable |

## Public Cached Prepare/Apply Surfaces

| public name | family | reuse role |
|---|---|---|
| `acb_mat.acb_mat_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `acb_mat.acb_mat_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `acb_mat.acb_mat_matvec_cached_apply_batch_fixed_prec` | `matrix` | prepared-plan apply |
| `acb_mat.acb_mat_matvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `acb_mat.acb_mat_matvec_cached_apply_batch_padded_prec` | `matrix` | prepared-plan apply |
| `acb_mat.acb_mat_matvec_cached_apply_jit` | `matrix` | prepared-plan apply |
| `acb_mat.acb_mat_matvec_cached_apply_prec` | `matrix` | prepared-plan apply |
| `acb_mat.acb_mat_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `acb_mat.acb_mat_matvec_cached_prepare_batch_fixed` | `matrix` | prepared-plan build |
| `acb_mat.acb_mat_matvec_cached_prepare_batch_fixed_prec` | `matrix` | prepared-plan build |
| `acb_mat.acb_mat_matvec_cached_prepare_batch_padded` | `matrix` | prepared-plan build |
| `acb_mat.acb_mat_matvec_cached_prepare_batch_padded_prec` | `matrix` | prepared-plan build |
| `acb_mat.acb_mat_matvec_cached_prepare_prec` | `matrix` | prepared-plan build |
| `acb_mat.acb_mat_rmatvec_cached_apply` | `matrix` | prepared-plan apply |
| `acb_mat.acb_mat_rmatvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `acb_mat.acb_mat_rmatvec_cached_apply_batch_fixed_prec` | `matrix` | prepared-plan apply |
| `acb_mat.acb_mat_rmatvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `acb_mat.acb_mat_rmatvec_cached_apply_batch_padded_prec` | `matrix` | prepared-plan apply |
| `acb_mat.acb_mat_rmatvec_cached_apply_jit` | `matrix` | prepared-plan apply |
| `acb_mat.acb_mat_rmatvec_cached_apply_prec` | `matrix` | prepared-plan apply |
| `acb_mat.acb_mat_rmatvec_cached_prepare` | `matrix` | prepared-plan build |
| `acb_mat.acb_mat_rmatvec_cached_prepare_batch_fixed` | `matrix` | prepared-plan build |
| `acb_mat.acb_mat_rmatvec_cached_prepare_batch_fixed_prec` | `matrix` | prepared-plan build |
| `acb_mat.acb_mat_rmatvec_cached_prepare_batch_padded` | `matrix` | prepared-plan build |
| `acb_mat.acb_mat_rmatvec_cached_prepare_batch_padded_prec` | `matrix` | prepared-plan build |
| `acb_mat.acb_mat_rmatvec_cached_prepare_prec` | `matrix` | prepared-plan build |
| `acb_mat_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `acb_mat_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `acb_mat_matvec_cached_apply_batch_fixed_prec` | `matrix` | prepared-plan apply |
| `acb_mat_matvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `acb_mat_matvec_cached_apply_batch_padded_prec` | `matrix` | prepared-plan apply |
| `acb_mat_matvec_cached_apply_jit` | `matrix` | prepared-plan apply |
| `acb_mat_matvec_cached_apply_prec` | `matrix` | prepared-plan apply |
| `acb_mat_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `acb_mat_matvec_cached_prepare_batch_fixed` | `matrix` | prepared-plan build |
| `acb_mat_matvec_cached_prepare_batch_fixed_prec` | `matrix` | prepared-plan build |
| `acb_mat_matvec_cached_prepare_batch_padded` | `matrix` | prepared-plan build |
| `acb_mat_matvec_cached_prepare_batch_padded_prec` | `matrix` | prepared-plan build |
| `acb_mat_matvec_cached_prepare_prec` | `matrix` | prepared-plan build |
| `acb_mat_rmatvec_cached_apply` | `matrix` | prepared-plan apply |
| `acb_mat_rmatvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `acb_mat_rmatvec_cached_apply_batch_fixed_prec` | `matrix` | prepared-plan apply |
| `acb_mat_rmatvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `acb_mat_rmatvec_cached_apply_batch_padded_prec` | `matrix` | prepared-plan apply |
| `acb_mat_rmatvec_cached_apply_jit` | `matrix` | prepared-plan apply |
| `acb_mat_rmatvec_cached_apply_prec` | `matrix` | prepared-plan apply |
| `acb_mat_rmatvec_cached_prepare` | `matrix` | prepared-plan build |
| `acb_mat_rmatvec_cached_prepare_batch_fixed` | `matrix` | prepared-plan build |
| `acb_mat_rmatvec_cached_prepare_batch_fixed_prec` | `matrix` | prepared-plan build |
| `acb_mat_rmatvec_cached_prepare_batch_padded` | `matrix` | prepared-plan build |
| `acb_mat_rmatvec_cached_prepare_batch_padded_prec` | `matrix` | prepared-plan build |
| `acb_mat_rmatvec_cached_prepare_prec` | `matrix` | prepared-plan build |
| `arb_mat.arb_mat_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `arb_mat.arb_mat_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `arb_mat.arb_mat_matvec_cached_apply_batch_fixed_prec` | `matrix` | prepared-plan apply |
| `arb_mat.arb_mat_matvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `arb_mat.arb_mat_matvec_cached_apply_batch_padded_prec` | `matrix` | prepared-plan apply |
| `arb_mat.arb_mat_matvec_cached_apply_jit` | `matrix` | prepared-plan apply |
| `arb_mat.arb_mat_matvec_cached_apply_prec` | `matrix` | prepared-plan apply |
| `arb_mat.arb_mat_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `arb_mat.arb_mat_matvec_cached_prepare_batch_fixed` | `matrix` | prepared-plan build |
| `arb_mat.arb_mat_matvec_cached_prepare_batch_fixed_prec` | `matrix` | prepared-plan build |
| `arb_mat.arb_mat_matvec_cached_prepare_batch_padded` | `matrix` | prepared-plan build |
| `arb_mat.arb_mat_matvec_cached_prepare_batch_padded_prec` | `matrix` | prepared-plan build |
| `arb_mat.arb_mat_matvec_cached_prepare_prec` | `matrix` | prepared-plan build |
| `arb_mat.arb_mat_rmatvec_cached_apply` | `matrix` | prepared-plan apply |
| `arb_mat.arb_mat_rmatvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `arb_mat.arb_mat_rmatvec_cached_apply_batch_fixed_prec` | `matrix` | prepared-plan apply |
| `arb_mat.arb_mat_rmatvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `arb_mat.arb_mat_rmatvec_cached_apply_batch_padded_prec` | `matrix` | prepared-plan apply |
| `arb_mat.arb_mat_rmatvec_cached_apply_jit` | `matrix` | prepared-plan apply |
| `arb_mat.arb_mat_rmatvec_cached_apply_prec` | `matrix` | prepared-plan apply |
| `arb_mat.arb_mat_rmatvec_cached_prepare` | `matrix` | prepared-plan build |
| `arb_mat.arb_mat_rmatvec_cached_prepare_batch_fixed` | `matrix` | prepared-plan build |
| `arb_mat.arb_mat_rmatvec_cached_prepare_batch_fixed_prec` | `matrix` | prepared-plan build |
| `arb_mat.arb_mat_rmatvec_cached_prepare_batch_padded` | `matrix` | prepared-plan build |
| `arb_mat.arb_mat_rmatvec_cached_prepare_batch_padded_prec` | `matrix` | prepared-plan build |
| `arb_mat.arb_mat_rmatvec_cached_prepare_prec` | `matrix` | prepared-plan build |
| `arb_mat_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `arb_mat_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `arb_mat_matvec_cached_apply_batch_fixed_prec` | `matrix` | prepared-plan apply |
| `arb_mat_matvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `arb_mat_matvec_cached_apply_batch_padded_prec` | `matrix` | prepared-plan apply |
| `arb_mat_matvec_cached_apply_jit` | `matrix` | prepared-plan apply |
| `arb_mat_matvec_cached_apply_prec` | `matrix` | prepared-plan apply |
| `arb_mat_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `arb_mat_matvec_cached_prepare_batch_fixed` | `matrix` | prepared-plan build |
| `arb_mat_matvec_cached_prepare_batch_fixed_prec` | `matrix` | prepared-plan build |
| `arb_mat_matvec_cached_prepare_batch_padded` | `matrix` | prepared-plan build |
| `arb_mat_matvec_cached_prepare_batch_padded_prec` | `matrix` | prepared-plan build |
| `arb_mat_matvec_cached_prepare_prec` | `matrix` | prepared-plan build |
| `arb_mat_rmatvec_cached_apply` | `matrix` | prepared-plan apply |
| `arb_mat_rmatvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `arb_mat_rmatvec_cached_apply_batch_fixed_prec` | `matrix` | prepared-plan apply |
| `arb_mat_rmatvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `arb_mat_rmatvec_cached_apply_batch_padded_prec` | `matrix` | prepared-plan apply |
| `arb_mat_rmatvec_cached_apply_jit` | `matrix` | prepared-plan apply |
| `arb_mat_rmatvec_cached_apply_prec` | `matrix` | prepared-plan apply |
| `arb_mat_rmatvec_cached_prepare` | `matrix` | prepared-plan build |
| `arb_mat_rmatvec_cached_prepare_batch_fixed` | `matrix` | prepared-plan build |
| `arb_mat_rmatvec_cached_prepare_batch_fixed_prec` | `matrix` | prepared-plan build |
| `arb_mat_rmatvec_cached_prepare_batch_padded` | `matrix` | prepared-plan build |
| `arb_mat_rmatvec_cached_prepare_batch_padded_prec` | `matrix` | prepared-plan build |
| `arb_mat_rmatvec_cached_prepare_prec` | `matrix` | prepared-plan build |
| `dft.dft_matvec_cached_apply_basic` | `core` | prepared-plan apply |
| `dft.dft_matvec_cached_apply_basic_jit` | `core` | prepared-plan apply |
| `dft.dft_matvec_cached_apply_basic_with_diagnostics` | `core` | prepared-plan apply |
| `dft.dft_matvec_cached_apply_batch_fixed_basic` | `core` | prepared-plan apply |
| `dft.dft_matvec_cached_apply_batch_fixed_basic_jit` | `core` | prepared-plan apply |
| `dft.dft_matvec_cached_apply_batch_fixed_point` | `core` | prepared-plan apply |
| `dft.dft_matvec_cached_apply_batch_fixed_point_jit` | `core` | prepared-plan apply |
| `dft.dft_matvec_cached_apply_batch_padded_basic` | `core` | prepared-plan apply |
| `dft.dft_matvec_cached_apply_batch_padded_point` | `core` | prepared-plan apply |
| `dft.dft_matvec_cached_apply_point` | `core` | prepared-plan apply |
| `dft.dft_matvec_cached_apply_point_jit` | `core` | prepared-plan apply |
| `dft.dft_matvec_cached_apply_point_with_diagnostics` | `core` | prepared-plan apply |
| `dft.dft_matvec_cached_prepare_basic` | `core` | prepared-plan build |
| `dft.dft_matvec_cached_prepare_point` | `core` | prepared-plan build |
| `dft_matvec_cached_apply_basic` | `core` | prepared-plan apply |
| `dft_matvec_cached_apply_basic_jit` | `core` | prepared-plan apply |
| `dft_matvec_cached_apply_basic_with_diagnostics` | `core` | prepared-plan apply |
| `dft_matvec_cached_apply_batch_fixed_basic` | `core` | prepared-plan apply |
| `dft_matvec_cached_apply_batch_fixed_basic_jit` | `core` | prepared-plan apply |
| `dft_matvec_cached_apply_batch_fixed_point` | `core` | prepared-plan apply |
| `dft_matvec_cached_apply_batch_fixed_point_jit` | `core` | prepared-plan apply |
| `dft_matvec_cached_apply_batch_padded_basic` | `core` | prepared-plan apply |
| `dft_matvec_cached_apply_batch_padded_point` | `core` | prepared-plan apply |
| `dft_matvec_cached_apply_point` | `core` | prepared-plan apply |
| `dft_matvec_cached_apply_point_jit` | `core` | prepared-plan apply |
| `dft_matvec_cached_apply_point_with_diagnostics` | `core` | prepared-plan apply |
| `dft_matvec_cached_prepare_basic` | `core` | prepared-plan build |
| `dft_matvec_cached_prepare_point` | `core` | prepared-plan build |
| `scb_block_mat.scb_block_mat_adjoint_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `scb_block_mat.scb_block_mat_adjoint_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_block_mat.scb_block_mat_adjoint_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `scb_block_mat.scb_block_mat_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `scb_block_mat.scb_block_mat_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_block_mat.scb_block_mat_matvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `scb_block_mat.scb_block_mat_matvec_cached_apply_with_diagnostics` | `matrix` | prepared-plan apply |
| `scb_block_mat.scb_block_mat_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `scb_block_mat.scb_block_mat_rmatvec_cached_apply` | `matrix` | prepared-plan apply |
| `scb_block_mat.scb_block_mat_rmatvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_block_mat.scb_block_mat_rmatvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `scb_block_mat.scb_block_mat_rmatvec_cached_prepare` | `matrix` | prepared-plan build |
| `scb_block_mat_adjoint_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `scb_block_mat_adjoint_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_block_mat_adjoint_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `scb_block_mat_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `scb_block_mat_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_block_mat_matvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `scb_block_mat_matvec_cached_apply_with_diagnostics` | `matrix` | prepared-plan apply |
| `scb_block_mat_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `scb_block_mat_rmatvec_cached_apply` | `matrix` | prepared-plan apply |
| `scb_block_mat_rmatvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_block_mat_rmatvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `scb_block_mat_rmatvec_cached_prepare` | `matrix` | prepared-plan build |
| `scb_mat.scb_mat_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `scb_mat.scb_mat_matvec_cached_apply_basic` | `matrix` | prepared-plan apply |
| `scb_mat.scb_mat_matvec_cached_apply_basic_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_mat.scb_mat_matvec_cached_apply_basic_batch_padded` | `matrix` | prepared-plan apply |
| `scb_mat.scb_mat_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_mat.scb_mat_matvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `scb_mat.scb_mat_matvec_cached_apply_jit` | `matrix` | prepared-plan apply |
| `scb_mat.scb_mat_matvec_cached_apply_with_diagnostics` | `matrix` | prepared-plan apply |
| `scb_mat.scb_mat_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `scb_mat.scb_mat_matvec_cached_prepare_basic` | `matrix` | prepared-plan build |
| `scb_mat.scb_mat_rmatvec_cached_apply` | `matrix` | prepared-plan apply |
| `scb_mat.scb_mat_rmatvec_cached_apply_basic` | `matrix` | prepared-plan apply |
| `scb_mat.scb_mat_rmatvec_cached_apply_basic_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_mat.scb_mat_rmatvec_cached_apply_basic_batch_padded` | `matrix` | prepared-plan apply |
| `scb_mat.scb_mat_rmatvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_mat.scb_mat_rmatvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `scb_mat.scb_mat_rmatvec_cached_apply_jit` | `matrix` | prepared-plan apply |
| `scb_mat.scb_mat_rmatvec_cached_prepare` | `matrix` | prepared-plan build |
| `scb_mat.scb_mat_rmatvec_cached_prepare_basic` | `matrix` | prepared-plan build |
| `scb_mat_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `scb_mat_matvec_cached_apply_basic` | `matrix` | prepared-plan apply |
| `scb_mat_matvec_cached_apply_basic_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_mat_matvec_cached_apply_basic_batch_padded` | `matrix` | prepared-plan apply |
| `scb_mat_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_mat_matvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `scb_mat_matvec_cached_apply_jit` | `matrix` | prepared-plan apply |
| `scb_mat_matvec_cached_apply_with_diagnostics` | `matrix` | prepared-plan apply |
| `scb_mat_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `scb_mat_matvec_cached_prepare_basic` | `matrix` | prepared-plan build |
| `scb_mat_rmatvec_cached_apply` | `matrix` | prepared-plan apply |
| `scb_mat_rmatvec_cached_apply_basic` | `matrix` | prepared-plan apply |
| `scb_mat_rmatvec_cached_apply_basic_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_mat_rmatvec_cached_apply_basic_batch_padded` | `matrix` | prepared-plan apply |
| `scb_mat_rmatvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_mat_rmatvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `scb_mat_rmatvec_cached_apply_jit` | `matrix` | prepared-plan apply |
| `scb_mat_rmatvec_cached_prepare` | `matrix` | prepared-plan build |
| `scb_mat_rmatvec_cached_prepare_basic` | `matrix` | prepared-plan build |
| `scb_vblock_mat.scb_vblock_mat_adjoint_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `scb_vblock_mat.scb_vblock_mat_adjoint_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_vblock_mat.scb_vblock_mat_adjoint_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `scb_vblock_mat.scb_vblock_mat_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `scb_vblock_mat.scb_vblock_mat_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_vblock_mat.scb_vblock_mat_matvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `scb_vblock_mat.scb_vblock_mat_matvec_cached_apply_with_diagnostics` | `matrix` | prepared-plan apply |
| `scb_vblock_mat.scb_vblock_mat_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `scb_vblock_mat.scb_vblock_mat_rmatvec_cached_apply` | `matrix` | prepared-plan apply |
| `scb_vblock_mat.scb_vblock_mat_rmatvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_vblock_mat.scb_vblock_mat_rmatvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `scb_vblock_mat.scb_vblock_mat_rmatvec_cached_prepare` | `matrix` | prepared-plan build |
| `scb_vblock_mat_adjoint_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `scb_vblock_mat_adjoint_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_vblock_mat_adjoint_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `scb_vblock_mat_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `scb_vblock_mat_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_vblock_mat_matvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `scb_vblock_mat_matvec_cached_apply_with_diagnostics` | `matrix` | prepared-plan apply |
| `scb_vblock_mat_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `scb_vblock_mat_rmatvec_cached_apply` | `matrix` | prepared-plan apply |
| `scb_vblock_mat_rmatvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `scb_vblock_mat_rmatvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `scb_vblock_mat_rmatvec_cached_prepare` | `matrix` | prepared-plan build |
| `srb_block_mat.srb_block_mat_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `srb_block_mat.srb_block_mat_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `srb_block_mat.srb_block_mat_matvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `srb_block_mat.srb_block_mat_matvec_cached_apply_with_diagnostics` | `matrix` | prepared-plan apply |
| `srb_block_mat.srb_block_mat_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `srb_block_mat.srb_block_mat_rmatvec_cached_apply` | `matrix` | prepared-plan apply |
| `srb_block_mat.srb_block_mat_rmatvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `srb_block_mat.srb_block_mat_rmatvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `srb_block_mat.srb_block_mat_rmatvec_cached_prepare` | `matrix` | prepared-plan build |
| `srb_block_mat_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `srb_block_mat_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `srb_block_mat_matvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `srb_block_mat_matvec_cached_apply_with_diagnostics` | `matrix` | prepared-plan apply |
| `srb_block_mat_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `srb_block_mat_rmatvec_cached_apply` | `matrix` | prepared-plan apply |
| `srb_block_mat_rmatvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `srb_block_mat_rmatvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `srb_block_mat_rmatvec_cached_prepare` | `matrix` | prepared-plan build |
| `srb_mat.srb_mat_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `srb_mat.srb_mat_matvec_cached_apply_basic` | `matrix` | prepared-plan apply |
| `srb_mat.srb_mat_matvec_cached_apply_basic_batch_fixed` | `matrix` | prepared-plan apply |
| `srb_mat.srb_mat_matvec_cached_apply_basic_batch_padded` | `matrix` | prepared-plan apply |
| `srb_mat.srb_mat_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `srb_mat.srb_mat_matvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `srb_mat.srb_mat_matvec_cached_apply_jit` | `matrix` | prepared-plan apply |
| `srb_mat.srb_mat_matvec_cached_apply_with_diagnostics` | `matrix` | prepared-plan apply |
| `srb_mat.srb_mat_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `srb_mat.srb_mat_matvec_cached_prepare_basic` | `matrix` | prepared-plan build |
| `srb_mat.srb_mat_rmatvec_cached_apply` | `matrix` | prepared-plan apply |
| `srb_mat.srb_mat_rmatvec_cached_apply_basic` | `matrix` | prepared-plan apply |
| `srb_mat.srb_mat_rmatvec_cached_apply_basic_batch_fixed` | `matrix` | prepared-plan apply |
| `srb_mat.srb_mat_rmatvec_cached_apply_basic_batch_padded` | `matrix` | prepared-plan apply |
| `srb_mat.srb_mat_rmatvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `srb_mat.srb_mat_rmatvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `srb_mat.srb_mat_rmatvec_cached_apply_jit` | `matrix` | prepared-plan apply |
| `srb_mat.srb_mat_rmatvec_cached_prepare` | `matrix` | prepared-plan build |
| `srb_mat.srb_mat_rmatvec_cached_prepare_basic` | `matrix` | prepared-plan build |
| `srb_mat_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `srb_mat_matvec_cached_apply_basic` | `matrix` | prepared-plan apply |
| `srb_mat_matvec_cached_apply_basic_batch_fixed` | `matrix` | prepared-plan apply |
| `srb_mat_matvec_cached_apply_basic_batch_padded` | `matrix` | prepared-plan apply |
| `srb_mat_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `srb_mat_matvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `srb_mat_matvec_cached_apply_jit` | `matrix` | prepared-plan apply |
| `srb_mat_matvec_cached_apply_with_diagnostics` | `matrix` | prepared-plan apply |
| `srb_mat_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `srb_mat_matvec_cached_prepare_basic` | `matrix` | prepared-plan build |
| `srb_mat_rmatvec_cached_apply` | `matrix` | prepared-plan apply |
| `srb_mat_rmatvec_cached_apply_basic` | `matrix` | prepared-plan apply |
| `srb_mat_rmatvec_cached_apply_basic_batch_fixed` | `matrix` | prepared-plan apply |
| `srb_mat_rmatvec_cached_apply_basic_batch_padded` | `matrix` | prepared-plan apply |
| `srb_mat_rmatvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `srb_mat_rmatvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `srb_mat_rmatvec_cached_apply_jit` | `matrix` | prepared-plan apply |
| `srb_mat_rmatvec_cached_prepare` | `matrix` | prepared-plan build |
| `srb_mat_rmatvec_cached_prepare_basic` | `matrix` | prepared-plan build |
| `srb_vblock_mat.srb_vblock_mat_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `srb_vblock_mat.srb_vblock_mat_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `srb_vblock_mat.srb_vblock_mat_matvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `srb_vblock_mat.srb_vblock_mat_matvec_cached_apply_with_diagnostics` | `matrix` | prepared-plan apply |
| `srb_vblock_mat.srb_vblock_mat_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `srb_vblock_mat.srb_vblock_mat_rmatvec_cached_apply` | `matrix` | prepared-plan apply |
| `srb_vblock_mat.srb_vblock_mat_rmatvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `srb_vblock_mat.srb_vblock_mat_rmatvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `srb_vblock_mat.srb_vblock_mat_rmatvec_cached_prepare` | `matrix` | prepared-plan build |
| `srb_vblock_mat_matvec_cached_apply` | `matrix` | prepared-plan apply |
| `srb_vblock_mat_matvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `srb_vblock_mat_matvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `srb_vblock_mat_matvec_cached_apply_with_diagnostics` | `matrix` | prepared-plan apply |
| `srb_vblock_mat_matvec_cached_prepare` | `matrix` | prepared-plan build |
| `srb_vblock_mat_rmatvec_cached_apply` | `matrix` | prepared-plan apply |
| `srb_vblock_mat_rmatvec_cached_apply_batch_fixed` | `matrix` | prepared-plan apply |
| `srb_vblock_mat_rmatvec_cached_apply_batch_padded` | `matrix` | prepared-plan apply |
| `srb_vblock_mat_rmatvec_cached_prepare` | `matrix` | prepared-plan build |

## Canonical Example Evidence

| example | required reuse pattern |
|---|---|
| [examples/example_api_surface.ipynb](/examples/example_api_surface.ipynb) | bound API callable reuse and runtime parameterization |
| [examples/example_core_scalar_surface.ipynb](/examples/example_core_scalar_surface.ipynb) | bound point-batch reuse and stable dtype/mode policy |
| [examples/example_dense_matrix_surface.ipynb](/examples/example_dense_matrix_surface.ipynb) | cached dense prepare/apply reuse |
| [examples/example_sparse_matrix_surface.ipynb](/examples/example_sparse_matrix_surface.ipynb) | cached sparse and block/vblock prepare/apply reuse |
| [examples/example_matrix_free_operator_surface.ipynb](/examples/example_matrix_free_operator_surface.ipynb) | operator-plan and preconditioner reuse |
| [examples/example_fft_nufft_surface.ipynb](/examples/example_fft_nufft_surface.ipynb) | prepared transform plan reuse |
| [examples/example_gamma_family_surface.ipynb](/examples/example_gamma_family_surface.ipynb) | bound callable reuse with stable point-mode controls |
| [examples/example_barnes_double_gamma_surface.ipynb](/examples/example_barnes_double_gamma_surface.ipynb) | bound callable reuse with stable special-function controls |
| [examples/example_hypgeom_family_surface.ipynb](/examples/example_hypgeom_family_surface.ipynb) | bound point-batch reuse with stable hypergeom point/mode controls |

## Canonical Benchmark Evidence

| benchmark entrypoint | required reuse pattern |
|---|---|
| [benchmarks/matrix_surface_workbook.py](/benchmarks/matrix_surface_workbook.py) | dense, sparse, block, vblock, and matrix-free reuse comparisons |
| [benchmarks/run_hypgeom_benchmark_smoke.py](/benchmarks/run_hypgeom_benchmark_smoke.py) | fixed-shape padded hypgeom batch reuse |
| [benchmarks/special_function_hardening_benchmark.py](/benchmarks/special_function_hardening_benchmark.py) | cross-family hardening metrics for incomplete-tail, Barnes, and hypergeom surfaces |
| [benchmarks/benchmark_fft_nufft.py](/benchmarks/benchmark_fft_nufft.py) | prepared transform plan reuse |
| [benchmarks/benchmark_sparse_matrix_surface.py](/benchmarks/benchmark_sparse_matrix_surface.py) | cached sparse prepare/apply reuse |
| [benchmarks/benchmark_dense_matrix_surface.py](/benchmarks/benchmark_dense_matrix_surface.py) | dense repeated-call and cached apply behavior |
