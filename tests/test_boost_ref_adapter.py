from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER = REPO_ROOT / "benchmarks" / "boost_ref_adapter.py"


def _run(payload: dict) -> np.ndarray:
    cp = subprocess.run(
        [sys.executable, str(ADAPTER)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        cwd=REPO_ROOT,
    )
    return np.asarray(json.loads(cp.stdout), dtype=np.float64)


def test_boost_ref_adapter_unary_contract():
    out = _run({"function": "exp", "x": [0.0, 1.0]})
    assert out.shape == (2,)
    assert np.allclose(out, [1.0, np.e], rtol=1e-12, atol=1e-12)


def test_boost_ref_adapter_bivariate_contract():
    out = _run({"function": "besselj", "x": [0.2, 0.5], "nu": [0.5, 1.5], "z": [0.2, 0.5]})
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))


def test_native_boost_ref_adapter_contract():
    native = REPO_ROOT / "benchmarks" / "run_boost_ref_adapter.sh"
    if sys.platform.startswith("win") or not native.exists():
        return
    cp = subprocess.run(
        [str(native)],
        input=json.dumps({"function": "exp", "x": [0.0, 1.0]}),
        text=True,
        capture_output=True,
        check=True,
        cwd=REPO_ROOT,
    )
    out = np.asarray(json.loads(cp.stdout), dtype=np.float64)
    assert out.shape == (2,)
    assert np.allclose(out, [1.0, np.e], rtol=1e-12, atol=1e-12)
