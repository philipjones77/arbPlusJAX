Last updated: 2026-02-25T03:51:38Z

# Benchmarks

This harness compares arbPlusJAX against multiple backends and writes sweep outputs under `results/benchmarks/`.

## Backends
- C Arb reference: build the C reference libs and point `ARB_C_REF_DIR` at the build folder (see flint repo).
- SciPy: resolved via installed SciPy or `SCIPY_REPO`.
- JAX numpy: resolved via installed JAX or `JAX_REPO` (point‑only reference).
- mpmath: resolved via installed mpmath or `MPMATH_REPO` (arbitrary precision point reference).
- Mathematica: optional local (Windows/Linux) or cloud endpoint for point reference.

## Run
```bash
python tools/bench_harness.py --samples 2000 --seed 0 --c-ref-dir "C:\\path\\to\\flint\\build" \
  --sweep-samples 2000,5000 --sweep-seeds 0,1
```

Linux example:
```bash
python tools/bench_harness.py --samples 2000 --seed 0 --c-ref-dir "/path/to/flint/build" \
  --sweep-samples 2000,5000 --sweep-seeds 0,1
```

## Mathematica (local + cloud)

Environment variables:
- `WOLFRAM_WINDOWS_DIR` (e.g. `C:\Program Files\Wolfram Research\Wolfram\14.3`)
- `WOLFRAM_LINUX_DIR` (e.g. `/usr/local/Wolfram/Mathematica/14.3`)
- `WOLFRAM_CLOUD_API_KEY` (set locally, not stored in repo)
- `WOLFRAM_CLOUD_URL` (deployed Wolfram Cloud endpoint)

Example (PowerShell):
```powershell
$env:WOLFRAM_WINDOWS_DIR = "C:\Program Files\Wolfram Research\Wolfram\14.3"
$env:WOLFRAM_CLOUD_URL = "https://www.wolframcloud.com/obj/pajones/arbplusjax-bench"
```

Example (bash):
```bash
export WOLFRAM_LINUX_DIR="/usr/local/Wolfram/Mathematica/14.3"
export WOLFRAM_CLOUD_URL="https://www.wolframcloud.com/obj/pajones/arbplusjax-bench"
```

Status:
- Local Windows: validated via `wolframscript.exe` when the install dir is set.
- Cloud: not validated unless a run includes `mathematica_cloud` in `results/benchmarks`.

JAX batch timing:
```bash
python tools/bench_harness.py --jax-batch --samples 5000 --seed 0
```

JAX point kernel timing:
```bash
python tools/bench_harness.py --jax-point-batch --samples 5000 --seed 0
```

Loggamma compare (real + complex + branch‑cut stress):
```bash
python tools/benchmark_loggamma_compare.py --arb-repo "C:\Users\phili\OneDrive\Documents\GitHub\arbPlusJAX\stuff" --iters 2000 --seed 0 --range-lo 0.1 --range-hi 8.0 --imag-range 6.0 --rad 0.05 --mp-dps 50
```

## Outputs
Each sweep run writes:
- `summary.csv` and `summary.json`
- per-function error histograms (`*_err_hist.csv`) and detail stats (`*_detail.json`)
- `sweep_index.json` in the run root

## Recent runs

- `run_20260225T022435Z`: bessel suite, 5000 samples, warmup timing, updated bessel bounds (`results/benchmarks/run_20260225T022435Z`).
- `run_20260225T021959Z`: bessel suite, 5000 samples, warmup timing, asymptotic bessel eval (`results/benchmarks/run_20260225T021959Z`).
- `loggamma-compare-2026-02-25T03:42:10Z`: loggamma compare tool run (real+complex + branch-cut stress), see `results/runs.csv`.

## Notes
Containment is measured by testing whether JAX intervals are contained in the C Arb intervals, and whether point outputs fall inside the C intervals.

Mode names used by the harness:
- `point`, `basic`, `adaptive`, `rigorous`

Bessel sampling:
- For `besselj/bessely/besseli/besselk`, the harness samples positive `z` to avoid complex-valued outputs when `nu` is non-integer.

## CLI options

Common flags:
- `--samples <int>`: number of samples per run.
- `--seed <int>`: RNG seed.
- `--sweep-samples a,b,c`: run multiple sample sizes.
- `--sweep-seeds a,b,c`: run multiple seeds.
- `--dps <int>`: decimal precision for mpmath.
- `--prec-bits <int>`: precision for interval inflation.
- `--functions f1,f2,...`: restrict to specific functions.
- `--outdir <path>`: output directory root.
- `--c-ref-dir <path>`: C ref build directory (optional; auto-detect if omitted).

JAX timing modes:
- `--jax-batch`: JIT one batched call for interval kernels (basic/adaptive/rigorous).
- `--jax-point-batch`: JIT one batched call for point-only kernels (`jax_point` backend).
- `--jax-warmup`: warm up JAX kernels before timing (excludes compile cost).

Mathematica:
- `--wolfram-cloud-url <url>`: cloud endpoint (or set `WOLFRAM_CLOUD_URL`).
- `--wolfram-windows-dir <path>`: Windows install dir (or set `WOLFRAM_WINDOWS_DIR`).
- `--wolfram-linux-dir <path>`: Linux install dir (or set `WOLFRAM_LINUX_DIR`).
