#!/usr/bin/env bash
set -euo pipefail

# Linux/macOS wrapper
python tools/run_validation.py "$@"
