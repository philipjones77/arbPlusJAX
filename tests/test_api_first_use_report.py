from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from arbplusjax import import_tiers
from tools import api_first_use_report as afur


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str) -> dict[str, object]:
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


def test_api_first_use_report_is_current() -> None:
    path = REPO_ROOT / "docs" / "reports" / "api_first_use_inventory.md"
    assert path.read_text(encoding="utf-8") == afur.render()


def test_first_use_module_budgets_hold() -> None:
    point_core = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import api
api.eval_point("exp", jnp.array([0.5], dtype=jnp.float32), dtype="float32")
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "mods": mods}))
"""
    )
    point_matrix = _run_python(
        """
import json
import sys
from arbplusjax import api
api.eval_point("arb_mat_zero", 2, dtype="float32")
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "mods": mods}))
"""
    )
    point_matrix_plan_prepare = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import api, double_interval as di
api.eval_point(
    "arb_mat_matvec_cached_prepare",
    di.interval(jnp.eye(2, dtype=jnp.float32), jnp.eye(2, dtype=jnp.float32)),
    dtype="float32",
)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "mods": mods}))
"""
    )
    point_matrix_plan_apply = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import api, double_interval as di
plan = api.eval_point(
    "arb_mat_matvec_cached_prepare",
    di.interval(jnp.eye(2, dtype=jnp.float32), jnp.eye(2, dtype=jnp.float32)),
    dtype="float32",
)
api.eval_point(
    "arb_mat_matvec_cached_apply",
    plan,
    di.interval(jnp.array([1.0, 2.0], dtype=jnp.float32), jnp.array([1.0, 2.0], dtype=jnp.float32)),
    dtype="float32",
)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "mods": mods}))
"""
    )
    tail = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import api
api.tail_integral(lambda t: jnp.exp(-t), jnp.array(1.0, dtype=jnp.float32))
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "mods": mods}))
"""
    )

    assert point_core["count"] <= import_tiers.POINT_CORE_FIRST_USE_MODULE_BUDGET
    assert point_matrix["count"] <= import_tiers.POINT_MATRIX_DENSE_FIRST_USE_MODULE_BUDGET
    assert point_matrix_plan_prepare["count"] <= import_tiers.POINT_MATRIX_PLAN_PREPARE_FIRST_USE_MODULE_BUDGET
    assert point_matrix_plan_apply["count"] <= import_tiers.POINT_MATRIX_PLAN_APPLY_FIRST_USE_MODULE_BUDGET
    assert tail["count"] <= import_tiers.TAIL_FIRST_USE_MODULE_BUDGET
    assert "arbplusjax.point_wrappers_core" in point_core["mods"]
    assert "arbplusjax.point_wrappers_matrix_dense" in point_matrix["mods"]
    assert "arbplusjax.point_wrappers_matrix_plans" in point_matrix_plan_prepare["mods"]
    assert "arbplusjax.point_wrappers_matrix_plans" in point_matrix_plan_apply["mods"]
    assert "arbplusjax.special.tail_acceleration" in tail["mods"]
