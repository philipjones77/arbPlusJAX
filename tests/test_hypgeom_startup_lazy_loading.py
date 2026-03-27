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


def test_importing_api_does_not_eagerly_import_hypgeom_modules() -> None:
    payload = _run_python(
        """
import json
import sys
from arbplusjax import api
print(json.dumps({
    "hypgeom": "arbplusjax.hypgeom" in sys.modules,
    "point_wrappers_hypgeom": "arbplusjax.point_wrappers_hypgeom" in sys.modules,
    "hypgeom_wrappers": "arbplusjax.hypgeom_wrappers" in sys.modules,
    "boost_hypgeom": "arbplusjax.boost_hypgeom" in sys.modules,
}))
"""
    )
    assert payload == {
        "hypgeom": False,
        "point_wrappers_hypgeom": False,
        "hypgeom_wrappers": False,
        "boost_hypgeom": False,
    }


def test_point_hypgeom_batch_path_stays_off_interval_hypgeom_modules() -> None:
    payload = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import api

_ = api.bind_point_batch_jit("hypgeom.arb_hypgeom_1f1", dtype="float32", pad_to=4)(
    jnp.array([1.0, 2.0], dtype=jnp.float32),
    jnp.array([2.0, 3.0], dtype=jnp.float32),
    jnp.array([0.5, 1.5], dtype=jnp.float32),
)
print(json.dumps({
    "point_wrappers_hypgeom": "arbplusjax.point_wrappers_hypgeom" in sys.modules,
    "hypgeom_wrappers": "arbplusjax.hypgeom_wrappers" in sys.modules,
    "boost_hypgeom": "arbplusjax.boost_hypgeom" in sys.modules,
}))
"""
    )
    assert payload == {
        "point_wrappers_hypgeom": True,
        "hypgeom_wrappers": False,
        "boost_hypgeom": False,
    }
