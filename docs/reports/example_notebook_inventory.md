Last updated: 2026-03-24T12:30:00Z

# Example Notebook Inventory

This report records the current `example_*.ipynb` coverage in the repo and the
remaining notebook gaps by functionality group.

Policy lives in:

- [example_notebook_standard.md](/docs/standards/example_notebook_standard.md)

Current notebook root:

- `/examples`

## Current Notebooks

- `example_all_modes_sweep.ipynb`
- `example_api_surface.ipynb`
- `example_bessel_modes_sweep.ipynb`
- `example_calc_modes_demo.ipynb`
- `example_core_modes_sweep.ipynb`
- `example_core_scalar_surface.ipynb`
- `example_dense_matrix_surface.ipynb`
- `example_dense_structured_spectral.ipynb`
- `example_dirichlet_surface.ipynb`
- `example_fft_nufft_surface.ipynb`
- `example_gamma_family_surface.ipynb`
- `example_hypgeom_robust_modes_sweep.ipynb`
- `example_large_sweeps_progress.ipynb`
- `example_matrix_free_operator_surface.ipynb`
- `example_sparse_matrix_surface.ipynb`
- `example_special_modes_sweep.ipynb`
- `example_barnes_double_gamma_surface.ipynb`

## Coverage By Functionality Group

### Core / scalar

Current notebook coverage:

- `example_core_modes_sweep.ipynb`
- `example_core_scalar_surface.ipynb`

Status:

- covered

### Special functions

Current notebook coverage:

- `example_special_modes_sweep.ipynb`
- `example_bessel_modes_sweep.ipynb`
- `example_hypgeom_robust_modes_sweep.ipynb`
- `example_dirichlet_surface.ipynb`
- `example_gamma_family_surface.ipynb`
- `example_barnes_double_gamma_surface.ipynb`

Status:

- covered

Notes:

- Bessel, hypergeometric, Dirichlet, gamma, and Barnes/double-gamma families now have dedicated notebooks.
- Tail-acceleration-specific and narrower per-subfamily notebooks can still be added later without changing category coverage.

### Dense matrix

Current notebook coverage:

- `example_dense_matrix_surface.ipynb`
- `example_dense_structured_spectral.ipynb`

Status:

- covered

### Sparse / block / vblock matrix

Current notebook coverage:

- `example_sparse_matrix_surface.ipynb`

Status:

- covered

### Matrix-free / operator

Current notebook coverage:

- `example_matrix_free_operator_surface.ipynb`
- related non-notebook example: `example_matrix_free_adjoints.py`

Status:

- covered

### Transforms

Current notebook coverage:

- `example_fft_nufft_surface.ipynb`

Status:

- covered

### API / runtime routing

Current notebook coverage:

- `example_api_surface.ipynb`

Status:

- covered

### Analytic / algebraic families

Current notebook coverage:

- `example_calc_modes_demo.ipynb`

Status:

- partially covered

Notes:

- Modular, elliptic, Bernoulli, partitions, and related families still do not have dedicated family notebooks.

## Required Notebook Content Checklist

Each functionality-group notebook should include:

- object or input instantiation
- public operation usage
- production-calling guidance:
  - binder reuse, cached plan reuse, or both where relevant
  - stable dtype/mode/precision choices
  - optional padding/chunking or other anti-recompile controls where relevant
- parameter/value sweeps
- summarized test/validation results
- summarized benchmark results
- comparisons to available benchmark/reference software
- graphs for relevant benchmark/value sweeps
- optional full diagnostics section or artifact links
- benchmark-extension guidance for adjacent functions in the same family

## Missing High-Priority Notebooks

- no current top-level notebook gaps in the canonical covered groups
- modular, elliptic, Bernoulli, partitions, and related analytic/algebraic families remain optional follow-on dedicated family notebooks

## Notes

- The current notebook set is strongest for modes, dense matrices, Bessel, and hypergeometric coverage.
- The repo now has canonical notebooks for the main top-level functionality groups.
- Some narrower subfamily-specific notebooks are still optional follow-on work rather than top-level gaps.
- Benchmarks, comparisons, and plots should be summarized in notebooks, but full raw artifacts should remain under `experiments/benchmarks/`.
