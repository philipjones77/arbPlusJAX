#!/usr/bin/env bash
set -euo pipefail

# Linux bootstrap for local development and GPU runs.
# Usage examples:
#   bash tools/setup_linux_gpu.sh
#   INSTALL_SYSTEM_DEPS=1 bash tools/setup_linux_gpu.sh
#   INSTALL_GPU_JAX=1 bash tools/setup_linux_gpu.sh
#   INSTALL_GPU_JAX=1 JAX_GPU_EXTRAS='jax[cuda12]' bash tools/setup_linux_gpu.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}"
INSTALL_GPU_JAX="${INSTALL_GPU_JAX:-0}"
JAX_GPU_EXTRAS="${JAX_GPU_EXTRAS:-jax[cuda12]}"

if [[ "$INSTALL_SYSTEM_DEPS" == "1" ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y build-essential cmake git libgmp-dev libmpfr-dev libflint-dev
  else
    echo "INSTALL_SYSTEM_DEPS=1 is only implemented for apt-based systems." >&2
    exit 1
  fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install -U pytest mpmath scipy

if [[ "$INSTALL_GPU_JAX" == "1" ]]; then
  # GPU wheels can change over time; this uses the modern extras interface.
  python -m pip install -U "$JAX_GPU_EXTRAS"
else
  python -m pip install -U jax jaxlib
fi

cat <<MSG

Linux environment is ready.

Activate:
  source "$VENV_DIR/bin/activate"

Optional parity setup (if C refs are built):
  export ARB_C_REF_DIR="$ROOT_DIR/stuff/migration/c_chassis/build"
  export LD_LIBRARY_PATH="\$ARB_C_REF_DIR:\$LD_LIBRARY_PATH"
  export ARBPLUSJAX_RUN_PARITY=1

Verify JAX runtime/device:
  python tools/check_jax_runtime.py --quick-bench

MSG
