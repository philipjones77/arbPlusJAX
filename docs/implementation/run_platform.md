Last updated: 2026-03-14T00:00:00Z

# Multi-Environment Run Platform

This repo is not being treated as a one-way migration target. It remains the active workspace, while also serving as a repeatable run platform for:

- Windows local development
- Linux local development and long-running jobs
- Google Colab exploratory and GPU-heavy runs

The rule is simple:

- `tests/` owns correctness
- `benchmarks/` owns performance and cross-backend comparison
- `tools/run_test_harness.py` owns test orchestration
- `benchmarks/run_benchmarks.py` and `benchmarks/run_harness_profile.py` own benchmark orchestration

## Environment matrix

| Environment | Primary use | Test harness | Benchmark harness |
|---|---|---|---|
| Windows | development, parity, local validation | supported | supported |
| Linux | development, CI-style validation, long runs | supported | supported |
| Google Colab | GPU runs, notebook sweeps, benchmark batches | supported | supported |

## VS Code root

The VS Code explorer root should be the repository root, [arbplusJAX](/home/phili/projects/arbplusJAX), not [docs](/home/phili/projects/arbplusJAX/docs).

Workspace files:

- `arbPlusJAX.code-workspace`: not currently present in this repo
- [arbPlusJAX-linux.code-workspace](/home/phili/projects/arbplusJAX/arbPlusJAX-linux.code-workspace): Linux-specific defaults
- [arbPlusJAX-windows.code-workspace](/home/phili/projects/arbplusJAX/arbPlusJAX-windows.code-workspace): Windows-specific defaults

Recommendation:

- use the Linux workspace on Linux
- use the Windows workspace on Windows
- keep Google Colab outside the VS Code workspace model and use the harness scripts directly

## Recommended workflows

Windows PowerShell:

```powershell
python .\tools\run_test_harness.py --profile chassis --jax-mode cpu
python .\tools\run_benchmarks.py --profile quick
```

Linux:

```bash
python tools/run_test_harness.py --profile chassis --jax-mode cpu
python benchmarks/run_benchmarks.py --profile quick
```

Google Colab:

```bash
!python /content/arbplusJAX/tools/check_jax_runtime.py --quick-bench
!python /content/arbplusJAX/tools/run_test_harness.py --profile chassis --jax-mode gpu
!python /content/arbplusJAX/benchmarks/run_benchmarks.py --profile quick
```

## Test and benchmark contract

- Chassis tests should stay runnable without Arb C references.
- Parity tests may require `ARB_C_REF_DIR` and platform-specific dynamic-library setup.
- Benchmark smoke tests should remain lightweight and invocable through pytest.
- Full benchmark sweeps should stay outside the normal test harness so long runs can be scheduled independently.
- When `tools/run_test_harness.py` is used with `--outdir`, and for every `benchmarks/bench_harness.py` run, both surfaces now emit the same `runtime_manifest.json` schema so OS, local/WSL/Colab, Python, JAX backend, and key env vars are recorded consistently.

## Source-tree execution rule

- Tests, benchmarks, and notebooks should import `arbplusjax` from `src/arbplusjax` in the current workspace.
- Do not rely on a separately installed wheel or editable package copy when validating the active repo state.
- Repo-root, `tests/`, `examples/`, `experiments/`, and `benchmarks/` source-tree bootstrap files should keep `src/` on `sys.path` for normal local execution.

## Matrix-specific note

Current matrix coverage is dense and banded, not general sparse. The matrix harness coverage currently centers on:

- dense `matmul`
- dense `matvec`
- cached dense matvec prepare/apply
- banded matvec
- factorization and solve chassis

If sparse work is added later, it should land as a separate matrix family with its own harness profile instead of being folded into the dense matrix path.
