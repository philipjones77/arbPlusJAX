from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


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


def test_matrix_free_core_import_and_operator_boundary_stay_off_solver_modules() -> None:
    import_only = _run_python(
        """
import json
import sys
from arbplusjax import matrix_free_core as mfc
del mfc
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"mods": mods}))
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
print(json.dumps({"mods": mods}))
"""
    )

    for payload in (import_only, operator_apply):
        assert "arbplusjax.matrix_free_core" in payload["mods"]
        assert "arbplusjax.matrix_free_krylov" not in payload["mods"]
        assert "arbplusjax.matrix_free_adjoint" not in payload["mods"]
        assert "arbplusjax.iterative_solvers" not in payload["mods"]
        assert "arbplusjax.matfree_adjoints" not in payload["mods"]


def test_matrix_free_solver_boundaries_load_only_requested_runtime_layer() -> None:
    krylov = _run_python(
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
print(json.dumps({"mods": mods}))
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
print(json.dumps({"mods": mods}))
"""
    )
    explicit_adjoints = _run_python(
        """
import json
import sys
from arbplusjax import matrix_free_core as mfc
mfc.matfree_adjoints.cg_fixed_iterations(num_matvecs=8)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"mods": mods}))
"""
    )

    assert "arbplusjax.matrix_free_krylov" in krylov["mods"]
    assert "arbplusjax.iterative_solvers" in krylov["mods"]
    assert "arbplusjax.matrix_free_adjoint" not in krylov["mods"]
    assert "arbplusjax.matfree_adjoints" not in krylov["mods"]

    assert "arbplusjax.matrix_free_adjoint" in implicit_adjoint["mods"]
    assert "arbplusjax.iterative_solvers" in implicit_adjoint["mods"]
    assert "arbplusjax.matrix_free_krylov" not in implicit_adjoint["mods"]
    assert "arbplusjax.matfree_adjoints" not in implicit_adjoint["mods"]

    assert "arbplusjax.matfree_adjoints" in explicit_adjoints["mods"]
