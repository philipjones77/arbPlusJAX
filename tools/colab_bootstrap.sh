#!/usr/bin/env bash
set -euo pipefail

# Standard Colab bootstrap for arbPlusJAX.
# Intended to run inside a Colab notebook cell:
#   !bash tools/colab_bootstrap.sh

REPO_DIR="${1:-$(pwd)}"
INSTALL_GPU_JAX="${INSTALL_GPU_JAX:-1}"
JAX_GPU_EXTRAS="${JAX_GPU_EXTRAS:-jax[cuda12]}"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "Expected a git checkout at: $REPO_DIR" >&2
  echo "Clone your repo first, then rerun this script." >&2
  exit 1
fi

cd "$REPO_DIR"
python -m pip install --upgrade pip setuptools wheel

if [[ "$INSTALL_GPU_JAX" == "1" ]]; then
  python -m pip install -U "$JAX_GPU_EXTRAS"
else
  python -m pip install -U jax jaxlib
fi

python -m pip install -e .
python -m pip install -U pytest scipy mpmath

# Reduce initial GPU memory grab in shared notebook runtimes.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Colab bootstrap complete at $REPO_DIR"
echo "Run: python tools/check_jax_runtime.py --quick-bench"
