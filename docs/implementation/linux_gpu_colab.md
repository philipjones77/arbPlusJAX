Last updated: 2026-03-05T00:00:00Z

# Linux + GPU + Colab Runbook

This runbook adds Linux/GPU and Colab workflows without removing Windows support. It is part of the multi-environment run platform, not a one-way migration plan.

## 1) Linux baseline (local dev)

From repo root:

```bash
bash tools/setup_linux_gpu.sh
source .venv/bin/activate
python tools/run_test_harness.py --profile chassis --jax-mode cpu
```

If you also need C parity dependencies on Ubuntu/Debian:

```bash
INSTALL_SYSTEM_DEPS=1 bash tools/setup_linux_gpu.sh
```

## 2) Linux C parity setup (Arb/FLINT refs)

Build refs (example):

```bash
cmake -S stuff/migration/c_chassis -B stuff/migration/c_chassis/build -DCMAKE_BUILD_TYPE=Release
cmake --build stuff/migration/c_chassis/build -j
```

Run parity:

```bash
export ARB_C_REF_DIR="$PWD/stuff/migration/c_chassis/build"
export LD_LIBRARY_PATH="$ARB_C_REF_DIR:$LD_LIBRARY_PATH"
python tools/run_test_harness.py --profile parity --jax-mode cpu
```

## 3) GPU verification and tuning

Quick runtime check:

```bash
python tools/check_jax_runtime.py --quick-bench
```

Recommended env for shared GPU machines:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Optional: cap allocator usage (for multitasking)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
```

Recommended benchmark pattern for stable GPU timings:

```bash
python tools/run_benchmarks.py --profile quick
```

The benchmark harness already supports batched JAX mode by default.

## 4) Colab standard workflow (large runs)

In Colab:

1. Runtime -> Change runtime type -> `T4/L4/A100 GPU`.
2. In a notebook cell:

```bash
%%bash
cd /content
# Replace REPO_URL with your real GitHub repo URL.
REPO_URL="https://github.com/<YOUR_ORG_OR_USER>/arbplusJAX.git" \
BRANCH="main" \
INSTALL_GPU_JAX=1 \
bash /content/arbplusJAX/tools/colab_bootstrap.sh /content/arbplusJAX
```

3. Validate GPU:

```bash
!python /content/arbplusJAX/tools/check_jax_runtime.py --quick-bench
```

4. Run larger sweeps and persist output to Drive:

```python
from google.colab import drive
from pathlib import Path
import shutil

drive.mount('/content/drive')
run_dir = Path('/content/arbplusJAX/experiments/benchmarks/results')
out_dir = Path('/content/drive/MyDrive/arbplusjax_runs')
out_dir.mkdir(parents=True, exist_ok=True)

# after benchmark run completes
for p in run_dir.glob('run_*'):
    dst = out_dir / p.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(p, dst)
```

5. Run correctness checks in Colab when needed:

```bash
!python /content/arbplusJAX/tools/run_test_harness.py --profile chassis --jax-mode gpu
!python /content/arbplusJAX/tools/run_test_harness.py --profile bench-smoke --jax-mode gpu
```

## 5) Windows compatibility note

These additions are additive only:
- Existing PowerShell commands in `README.md` and `docs/implementation/build.md` stay valid.
- CI now runs chassis tests on both Ubuntu and Windows.
