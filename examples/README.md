# Examples Notebook Suite

Notebook policy and coverage:

- [example_notebook_standard.md](/home/phili/projects/arbplusJAX/docs/standards/example_notebook_standard.md)
- [example_notebook_inventory.md](/home/phili/projects/arbplusJAX/docs/reports/example_notebook_inventory.md)
- [environment_portability_standard.md](/home/phili/projects/arbplusJAX/docs/standards/environment_portability_standard.md)

All notebooks in this folder are named `example_*.ipynb` and are designed for both Linux and Windows.

Non-notebook example helper files should also use the `example_` prefix when
they are part of the canonical example surface.

Default execution model:

- notebooks run against the repo source tree through `/src`
- installed-package execution is secondary, not the default notebook path

## Example-owned inputs and outputs

- `examples/inputs/`: tracked/shared example input roots
- `examples/outputs/`: retained example output roots

Each example should own its own named subfolder under both roots.

Examples:
- `examples/inputs/example_dense_matrix_surface/`
- `examples/outputs/example_dense_matrix_surface/`
- `examples/inputs/example_run_suite/`
- `examples/outputs/example_run_suite/`

Rules:
- do not scatter ad hoc files directly under `examples/inputs/` or `examples/outputs/`
- each example controls its own subfolder
- shared templates belong in the owning example folder, not duplicated across unrelated examples
- transient local-only output files should still be kept out of Git unless they are intentionally retained

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
- `example_matrix_free_adjoints.py`

These notebooks compare the four modes (`point`, `basic`, `adaptive`, `rigorous`) and report:
- timing (`time_ms`)
- peak memory of harness runs (`peak_rss_mb`)
- accuracy/containment versus C reference (`mean_abs_err`, `containment_rate`)

All run via `benchmarks/run_harness_profile.py` for consistent output.

For calc examples, note the layering:
- names like `acb_calc_integrate_line`, `acb_calc_integrate_gl_auto_deg`, and `acb_calc_integrate_taylor` are different numerical methods
- the four modes are still `point`, `basic`, `adaptive`, and `rigorous`
- mode selection is applied through `calc_wrappers` / `api`, not by the calc method name itself

For dense matrix examples:
- `example_dense_matrix_surface.ipynb` covers dense solve, cached matvec, and structured SPD / HPD solve-plan reuse
- `example_dense_structured_spectral.ipynb` covers dense symmetric / Hermitian eigendecomposition and structured solve midpoint behavior

For matrix-free adjoint examples:
- `example_matrix_free_adjoints.py` demonstrates direct scripted use of the matrix-free custom-adjoint surfaces outside the notebook flow

Dense matrix documentation:
- implementation overview: [dense_matrix_tranche.md](/home/phili/projects/arbplusJAX/docs/implementation/dense_matrix_tranche.md)
- practical runbook: [dense_matrices.md](/home/phili/projects/arbplusJAX/docs/practical/dense_matrices.md)

## Scripted suite runs

Use:

```bash
python examples/example_run_suite.py --config examples/inputs/example_run_suite/example_run.json
```

Start from template:

```bash
cp examples/inputs/example_run_suite/example_run_template.json examples/inputs/example_run_suite/example_run.json
```
