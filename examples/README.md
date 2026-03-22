# Examples Notebook Suite

Notebook policy and coverage:

- [example_notebook_standard.md](/home/phili/projects/arbplusJAX/docs/standards/example_notebook_standard.md)
- [example_notebook_inventory.md](/home/phili/projects/arbplusJAX/docs/reports/example_notebook_inventory.md)
- [environment_portability_standard.md](/home/phili/projects/arbplusJAX/docs/standards/environment_portability_standard.md)

All notebooks in this folder are named `example_*.ipynb` and are designed for both Linux and Windows.

Default execution model:

- notebooks run against the repo source tree through `/src`
- installed-package execution is secondary, not the default notebook path

## Local run folders (not committed)

- `examples/_input/`: local JSON run configs
- `examples/_output/`: local run artifacts and summaries

Both are gitignored.

## Runtime selection (CPU vs GPU)

Each notebook prints:
- active Python interpreter
- active JAX mode (`cpu` or `gpu`)
- active JAX dtype (`float64` or `float32`)
- active backend/devices from `tools/check_jax_runtime.py`

Set mode before running a notebook cell sequence:
- Linux/macOS: `export JAX_MODE=cpu` or `export JAX_MODE=gpu`
- Windows PowerShell: `$env:JAX_MODE='cpu'` or `$env:JAX_MODE='gpu'`

Set dtype before running a notebook cell sequence:
- Linux/macOS: `export JAX_DTYPE=float64` or `export JAX_DTYPE=float32`
- Windows PowerShell: `$env:JAX_DTYPE='float64'` or `$env:JAX_DTYPE='float32'`

## C reference path

Each notebook uses `ARB_C_REF_DIR` when set.
Default fallback is `stuff/migration/c_chassis/build_linux_wsl`.

## Notebooks

- `example_core_modes_sweep.ipynb`
- `example_special_modes_sweep.ipynb`
- `example_bessel_modes_sweep.ipynb`
- `example_hypgeom_robust_modes_sweep.ipynb`
- `example_all_modes_sweep.ipynb`
- `example_large_sweeps_progress.ipynb`
- `example_calc_modes_demo.ipynb`
- `example_dense_matrix_surface.ipynb`
- `example_dense_structured_spectral.ipynb`

These notebooks compare the four modes (`point`, `basic`, `adaptive`, `rigorous`) and report:
- timing (`time_ms`)
- peak memory of harness runs (`peak_rss_mb`)
- accuracy/containment versus C reference (`mean_abs_err`, `containment_rate`)

All run via `tools/run_harness_profile.py` for consistent output.

For calc examples, note the layering:
- names like `acb_calc_integrate_line`, `acb_calc_integrate_gl_auto_deg`, and `acb_calc_integrate_taylor` are different numerical methods
- the four modes are still `point`, `basic`, `adaptive`, and `rigorous`
- mode selection is applied through `calc_wrappers` / `api`, not by the calc method name itself

For dense matrix examples:
- `example_dense_matrix_surface.ipynb` covers dense solve, cached matvec, and structured SPD / HPD solve-plan reuse
- `example_dense_structured_spectral.ipynb` covers dense symmetric / Hermitian eigendecomposition and structured solve midpoint behavior

Dense matrix documentation:
- implementation overview: [dense_matrix_tranche.md](/home/phili/projects/arbplusJAX/docs/implementation/dense_matrix_tranche.md)
- practical runbook: [dense_matrices.md](/home/phili/projects/arbplusJAX/docs/practical/dense_matrices.md)

## Scripted suite runs

Use:

```bash
python examples/example_run_suite.py --config examples/_input/example_run.json
```

Start from template:

```bash
cp examples/example_run_template.json examples/_input/example_run.json
```
