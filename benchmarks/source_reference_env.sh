#!/usr/bin/env bash
set -euo pipefail

REF_PREFIX="${ARBPLUSJAX_REF_PREFIX:-$HOME/.local/opt/arbplusjax_refs}"
FLINT_ROOT="${FLINT_ROOT:-$REF_PREFIX/flint/current}"
BOOST_ROOT="${BOOST_ROOT:-$REF_PREFIX/boost/current}"
BOOST_INCLUDEDIR="${BOOST_INCLUDEDIR:-$BOOST_ROOT/include}"
BOOST_LIBRARYDIR="${BOOST_LIBRARYDIR:-$BOOST_ROOT/lib}"
WOLFRAM_LINUX_DIR="${WOLFRAM_LINUX_DIR:-$HOME/Wolfram/14.3/Executables}"

export ARBPLUSJAX_REF_PREFIX="$REF_PREFIX"
export FLINT_ROOT
export BOOST_ROOT
export BOOST_INCLUDEDIR
export BOOST_LIBRARYDIR
export WOLFRAM_LINUX_DIR
export BOOST_REF_CMD="${BOOST_REF_CMD:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_boost_ref_adapter.sh}"

if [[ -d "$FLINT_ROOT/lib" ]]; then
  export LD_LIBRARY_PATH="$FLINT_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export LIBRARY_PATH="$FLINT_ROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi

if [[ -d "$FLINT_ROOT/include" ]]; then
  export CPATH="$FLINT_ROOT/include${CPATH:+:$CPATH}"
fi

if [[ -d "$BOOST_LIBRARYDIR" ]]; then
  export LD_LIBRARY_PATH="$BOOST_LIBRARYDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export LIBRARY_PATH="$BOOST_LIBRARYDIR${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi

if [[ -d "$BOOST_INCLUDEDIR" ]]; then
  export CPATH="$BOOST_INCLUDEDIR${CPATH:+:$CPATH}"
fi
