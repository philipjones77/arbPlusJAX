from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from arbplusjax import import_tiers
from tools import matrix_free_first_use_report as mffur


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


def test_matrix_free_first_use_report_is_current() -> None:
    path = REPO_ROOT / "docs" / "reports" / "matrix_free_first_use_inventory.md"
    assert path.read_text(encoding="utf-8") == mffur.render()


def test_matrix_free_first_use_budgets_hold() -> None:
    operator_create = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import matrix_free_core as mfc
mfc.dense_operator_plan(jnp.eye(2, dtype=jnp.float32), orientation="forward", algebra="jrb")
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "mods": mods}))
"""
    )
    operator_apply = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import matrix_free_core as mfc
plan = mfc.dense_operator_plan(jnp.eye(2, dtype=jnp.float32), orientation="forward", algebra="jrb")
mfc.operator_plan_apply(
    plan,
    jnp.array([1.0, 2.0], dtype=jnp.float32),
    midpoint_vector=jnp.asarray,
    sparse_bcoo_matvec=lambda *args, **kwargs: None,
    dtype=jnp.float32,
)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "mods": mods}))
"""
    )
    krylov_solve = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import matrix_free_core as mfc
plan = mfc.dense_operator_plan(
    jnp.array([[2.0, 0.0], [0.0, 3.0]], dtype=jnp.float64),
    orientation="forward",
    algebra="jrb",
)
rhs = jnp.array([1.0, 1.0], dtype=jnp.float64)
mfc.krylov_solve_midpoint(
    plan,
    rhs,
    solver="cg",
    midpoint_vector=jnp.asarray,
    lift_vector=jnp.asarray,
    sparse_bcoo_matvec=lambda *args, **kwargs: None,
    dtype=jnp.float64,
    tol=1e-6,
    maxiter=4,
)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "mods": mods}))
"""
    )
    implicit_adjoint = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import matrix_free_core as mfc
plan = mfc.dense_operator_plan(
    jnp.array([[2.0, 0.0], [0.0, 3.0]], dtype=jnp.float64),
    orientation="forward",
    algebra="jrb",
)
rhs = jnp.array([1.0, 1.0], dtype=jnp.float64)
mfc.implicit_krylov_solve_midpoint(
    plan,
    rhs,
    solver="cg",
    structured="symmetric",
    midpoint_vector=jnp.asarray,
    lift_vector=jnp.asarray,
    sparse_bcoo_matvec=lambda *args, **kwargs: None,
    dtype=jnp.float64,
    tol=1e-6,
    maxiter=4,
)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "mods": mods}))
"""
    )

    assert operator_create["count"] <= import_tiers.MATRIX_FREE_OPERATOR_CREATE_FIRST_USE_MODULE_BUDGET
    assert operator_apply["count"] <= import_tiers.MATRIX_FREE_OPERATOR_APPLY_FIRST_USE_MODULE_BUDGET
    assert krylov_solve["count"] <= import_tiers.MATRIX_FREE_KRYLOV_SOLVE_FIRST_USE_MODULE_BUDGET
    assert implicit_adjoint["count"] <= import_tiers.MATRIX_FREE_IMPLICIT_ADJOINT_FIRST_USE_MODULE_BUDGET

    assert "arbplusjax.matrix_free_core" in operator_create["mods"]
    assert "arbplusjax.matrix_free_krylov" not in operator_create["mods"]
    assert "arbplusjax.matrix_free_adjoint" not in operator_create["mods"]
    assert "arbplusjax.iterative_solvers" not in operator_create["mods"]
    assert "arbplusjax.matfree_adjoints" not in operator_create["mods"]

    assert "arbplusjax.matrix_free_core" in operator_apply["mods"]
    assert "arbplusjax.matrix_free_krylov" not in operator_apply["mods"]
    assert "arbplusjax.matrix_free_adjoint" not in operator_apply["mods"]
    assert "arbplusjax.iterative_solvers" not in operator_apply["mods"]
    assert "arbplusjax.matfree_adjoints" not in operator_apply["mods"]

    assert "arbplusjax.matrix_free_krylov" in krylov_solve["mods"]
    assert "arbplusjax.iterative_solvers" in krylov_solve["mods"]
    assert "arbplusjax.matrix_free_adjoint" not in krylov_solve["mods"]
    assert "arbplusjax.matfree_adjoints" not in krylov_solve["mods"]

    assert "arbplusjax.matrix_free_adjoint" in implicit_adjoint["mods"]
    assert "arbplusjax.iterative_solvers" in implicit_adjoint["mods"]
    assert "arbplusjax.matrix_free_krylov" not in implicit_adjoint["mods"]
    assert "arbplusjax.matfree_adjoints" not in implicit_adjoint["mods"]
