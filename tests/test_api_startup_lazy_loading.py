from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str) -> dict[str, bool]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def test_importing_api_does_not_eagerly_import_interval_wrappers() -> None:
    payload = _run_python(
        """
import json
import sys
from arbplusjax import api
print(json.dumps({
    "baseline_wrappers": "arbplusjax.baseline_wrappers" in sys.modules,
    "mat_wrappers": "arbplusjax.mat_wrappers" in sys.modules,
    "hypgeom_wrappers": "arbplusjax.hypgeom_wrappers" in sys.modules,
}))
"""
    )
    assert payload == {
        "baseline_wrappers": False,
        "mat_wrappers": False,
        "hypgeom_wrappers": False,
    }


def test_point_batch_call_stays_off_interval_wrapper_modules() -> None:
    payload = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import api
_ = api.eval_point_batch(
    "add",
    jnp.array([1.0, 2.0], dtype=jnp.float32),
    jnp.array([3.0, 4.0], dtype=jnp.float32),
    dtype="float32",
    pad_to=4,
)
print(json.dumps({
    "baseline_wrappers": "arbplusjax.baseline_wrappers" in sys.modules,
    "mat_wrappers": "arbplusjax.mat_wrappers" in sys.modules,
    "hypgeom_wrappers": "arbplusjax.hypgeom_wrappers" in sys.modules,
}))
"""
    )
    assert payload == {
        "baseline_wrappers": False,
        "mat_wrappers": False,
        "hypgeom_wrappers": False,
    }
