#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/benchmarks/native"
BUILD_DIR="${BOOST_REF_BUILD_DIR:-$ROOT_DIR/stuff/migration/boost_ref_adapter/build_linux_wsl}"
BIN_PATH="$BUILD_DIR/boost_ref_adapter"

needs_build=0
if [[ ! -x "$BIN_PATH" ]]; then
  needs_build=1
elif [[ "$SRC_DIR/boost_ref_adapter.cpp" -nt "$BIN_PATH" || "$SRC_DIR/CMakeLists.txt" -nt "$BIN_PATH" ]]; then
  needs_build=1
fi

if [[ "$needs_build" -eq 1 ]]; then
  cmake -S "$SRC_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >&2
  cmake --build "$BUILD_DIR" --config Release -j >&2
fi

exec "$BIN_PATH"
