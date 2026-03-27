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


def test_importing_api_does_not_eagerly_import_dense_interval_matrix_modules() -> None:
    payload = _run_python(
        """
import json
import sys
from arbplusjax import api
print(json.dumps({
    "arb_mat": "arbplusjax.arb_mat" in sys.modules,
    "acb_mat": "arbplusjax.acb_mat" in sys.modules,
    "mat_wrappers": "arbplusjax.mat_wrappers" in sys.modules,
    "mat_wrappers_dense": "arbplusjax.mat_wrappers_dense" in sys.modules,
    "mat_wrappers_plans": "arbplusjax.mat_wrappers_plans" in sys.modules,
}))
"""
    )
    assert payload == {
        "arb_mat": False,
        "acb_mat": False,
        "mat_wrappers": False,
        "mat_wrappers_dense": False,
        "mat_wrappers_plans": False,
    }


def test_point_dense_matrix_batch_path_stays_off_dense_interval_matrix_modules() -> None:
    payload = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import api
from arbplusjax import double_interval as di

dense = jnp.asarray(
    [
        [[3.0, 0.5], [0.5, 2.0]],
        [[4.0, -1.0], [-1.0, 3.5]],
        [[2.5, 0.25], [0.25, 2.25]],
        [[5.0, 0.75], [0.75, 4.0]],
    ],
    dtype=jnp.float64,
)
rhs = jnp.asarray(
    [
        [1.0, 2.0],
        [1.5, -0.5],
        [0.75, 0.25],
        [0.25, 1.25],
    ],
    dtype=jnp.float64,
)
dense_interval = di.interval(dense, dense)
rhs_interval = di.interval(rhs, rhs)
_ = api.bind_point_batch_jit("arb_mat_matvec", dtype="float64", pad_to=4)(dense_interval, rhs_interval)
print(json.dumps({
    "arb_mat": "arbplusjax.arb_mat" in sys.modules,
    "acb_mat": "arbplusjax.acb_mat" in sys.modules,
    "mat_wrappers": "arbplusjax.mat_wrappers" in sys.modules,
    "mat_wrappers_dense": "arbplusjax.mat_wrappers_dense" in sys.modules,
    "mat_wrappers_plans": "arbplusjax.mat_wrappers_plans" in sys.modules,
}))
"""
    )
    assert payload == {
        "arb_mat": False,
        "acb_mat": False,
        "mat_wrappers": False,
        "mat_wrappers_dense": False,
        "mat_wrappers_plans": False,
    }
