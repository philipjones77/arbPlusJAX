Last updated: 2026-03-22T00:00:00Z

# Example Notebook Inventory

This report records the current `example_*.ipynb` coverage in the repo and the
remaining notebook gaps by functionality group.

Policy lives in:

- [example_notebook_standard.md](/home/phili/projects/arbplusJAX/docs/standards/example_notebook_standard.md)

Current notebook root:

- `/examples`

## Current Notebooks

- `example_all_modes_sweep.ipynb`
- `example_bessel_modes_sweep.ipynb`
- `example_calc_modes_demo.ipynb`
- `example_core_modes_sweep.ipynb`
- `example_dense_matrix_surface.ipynb`
- `example_dense_structured_spectral.ipynb`
- `example_hypgeom_robust_modes_sweep.ipynb`
- `example_large_sweeps_progress.ipynb`
- `example_special_modes_sweep.ipynb`

## Coverage By Functionality Group

### Core / scalar

Current notebook coverage:

- `example_core_modes_sweep.ipynb`

Status:

- covered

### Special functions

Current notebook coverage:

- `example_special_modes_sweep.ipynb`
- `example_bessel_modes_sweep.ipynb`
- `example_hypgeom_robust_modes_sweep.ipynb`

Status:

- partially covered

Notes:

- Bessel and hypergeometric families have dedicated notebooks.
- Gamma, Barnes/double-gamma, Hankel, incomplete gamma, incomplete Bessel, and tail-acceleration-specific notebooks are still missing as dedicated group notebooks.

### Dense matrix

Current notebook coverage:

- `example_dense_matrix_surface.ipynb`
- `example_dense_structured_spectral.ipynb`

Status:

- covered

### Sparse / block / vblock matrix

Current notebook coverage:

- none

Status:

- missing dedicated notebook

Recommended additions:

- `example_sparse_matrix_surface.ipynb`
- `example_block_sparse_matrix_surface.ipynb`

### Matrix-free / operator

Current notebook coverage:

- no dedicated `example_*.ipynb` notebook yet

Related non-notebook example:

- `matfree_adjoints_examples.py`

Status:

- missing dedicated notebook

Recommended additions:

- `example_matrix_free_operator_surface.ipynb`
- `example_matrix_free_logdet_eigsh.ipynb`

### Transforms

Current notebook coverage:

- none

Status:

- missing dedicated notebook

Recommended additions:

- `example_fft_nufft_surface.ipynb`

### API / runtime routing

Current notebook coverage:

- partially covered indirectly by modes and calc notebooks

Status:

- missing dedicated notebook

Recommended additions:

- `example_api_surface.ipynb`

### Analytic / algebraic families

Current notebook coverage:

- `example_calc_modes_demo.ipynb`

Status:

- partially covered

Notes:

- Dirichlet, modular, elliptic, Bernoulli, partitions, and related families do not yet have dedicated `example_` notebooks.

## Required Notebook Content Checklist

Each functionality-group notebook should include:

- object or input instantiation
- public operation usage
- parameter/value sweeps
- summarized test/validation results
- summarized benchmark results
- comparisons to available benchmark/reference software
- graphs for relevant benchmark/value sweeps
- optional full diagnostics section or artifact links

## Missing High-Priority Notebooks

- `example_sparse_matrix_surface.ipynb`
- `example_matrix_free_operator_surface.ipynb`
- `example_fft_nufft_surface.ipynb`
- `example_api_surface.ipynb`
- `example_gamma_family_surface.ipynb`
- `example_barnes_double_gamma_surface.ipynb`

## Notes

- The current notebook set is strongest for modes, dense matrices, Bessel, and hypergeometric coverage.
- The repo does not yet have full notebook parity across all major functionality groups.
- Benchmarks, comparisons, and plots should be summarized in notebooks, but full raw artifacts should remain under `experiments/benchmarks/`.
