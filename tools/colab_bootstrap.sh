#!/usr/bin/env bash
set -euo pipefail

# Standard Colab bootstrap for arbPlusJAX.
# Intended to run inside a Colab notebook cell:
#   !bash tools/colab_bootstrap.sh

REPO_DIR="${1:-$(pwd)}"
INSTALL_GPU_JAX="${INSTALL_GPU_JAX:-1}"
JAX_GPU_EXTRAS="${JAX_GPU_EXTRAS:-jax[cuda12]}"
INSTALL_BENCHMARK_BASE="${INSTALL_BENCHMARK_BASE:-1}"
INSTALL_NUFFTAX="${INSTALL_NUFFTAX:-0}"
INSTALL_JAX_FINUFFT_CPU="${INSTALL_JAX_FINUFFT_CPU:-0}"
INSTALL_PETSC_SLEPC="${INSTALL_PETSC_SLEPC:-0}"

install_pkg() {
  python -m pip install -U "$@"
}

try_install_pkg() {
  if python -m pip install -U "$@"; then
    return 0
  fi
  echo "Optional install failed: $*" >&2
  return 1
}

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "Expected a git checkout at: $REPO_DIR" >&2
  echo "Clone your repo first, then rerun this script." >&2
  exit 1
fi

cd "$REPO_DIR"
install_pkg pip setuptools wheel

if [[ "$INSTALL_GPU_JAX" == "1" ]]; then
  install_pkg "$JAX_GPU_EXTRAS"
else
  install_pkg jax jaxlib
fi

python -m pip install -e .

if [[ "$INSTALL_BENCHMARK_BASE" == "1" ]]; then
  install_pkg pytest scipy mpmath
fi

if [[ "$INSTALL_NUFFTAX" == "1" ]]; then
  install_pkg nufftax
fi

if [[ "$INSTALL_JAX_FINUFFT_CPU" == "1" ]]; then
  install_pkg jax-finufft
fi

if [[ "$INSTALL_PETSC_SLEPC" == "1" ]]; then
  echo "Attempting optional PETSc/SLEPc Python stack install for Colab." >&2
  echo "This is best-effort only and may fail due to native build/runtime limits." >&2
  try_install_pkg petsc petsc4py slepc slepc4py || true
fi

# Reduce initial GPU memory grab in shared notebook runtimes.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Colab bootstrap complete at $REPO_DIR"
echo "INSTALL_BENCHMARK_BASE=$INSTALL_BENCHMARK_BASE"
echo "INSTALL_NUFFTAX=$INSTALL_NUFFTAX"
echo "INSTALL_JAX_FINUFFT_CPU=$INSTALL_JAX_FINUFFT_CPU"
echo "INSTALL_PETSC_SLEPC=$INSTALL_PETSC_SLEPC"
echo "Run: python tools/check_jax_runtime.py --quick-bench"
