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
    estimator_first_use = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import matrix_free_core as mfc
mfc.make_probe_estimate_statistics(jnp.array([1.0, 2.0], dtype=jnp.float32))
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "mods": mods}))
"""
    )
    contour_first_use = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import matrix_free_core as mfc
mfc.contour_integral_action_point(
    lambda shift, v: v / (jnp.array([2.0, 3.0], dtype=jnp.complex128) - shift),
    jnp.array([1.0, 2.0], dtype=jnp.float32),
    center=2.5 + 0.0j,
    radius=0.75,
    quadrature_order=16,
    node_weight_fn=lambda node: jnp.log(node) / (2.0j * jnp.pi),
)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "mods": mods}))
"""
    )
    real_slq_wrapper_first_use = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import jrb_mat
a = jrb_mat.jrb_mat_as_interval_matrix(jnp.array([[[2.0, 2.0], [0.0, 0.0]], [[0.0, 0.0], [3.0, 3.0]]], dtype=jnp.float64))
probes = jrb_mat.jrb_mat_as_interval_vector(jnp.array([[1.0, 1.0], [0.5, 0.5]], dtype=jnp.float64))[None, ...]
jrb_mat.jrb_mat_logdet_estimate_point(jrb_mat.jrb_mat_dense_operator(a), probes, steps=2)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "mods": mods}))
"""
    )
    complex_slq_wrapper_first_use = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import jcb_mat
a = jcb_mat._jcb_point_box(jnp.array([[2.0+0.0j, 0.0+0.0j], [0.0+0.0j, 3.0+0.0j]], dtype=jnp.complex128))
probes = jcb_mat._jcb_point_box(jnp.array([1.0+0.0j, 0.5+0.0j], dtype=jnp.complex128))[None, ...]
jcb_mat.jcb_mat_logdet_estimate_point(jcb_mat.jcb_mat_dense_operator(a), probes, steps=2)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "mods": mods}))
"""
    )
    real_hutchpp_wrapper_first_use = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import jrb_mat
def action_fn(v):
    return v
probes = jrb_mat.jrb_mat_as_interval_vector(jnp.array([[1.0, 1.0], [0.5, 0.5]], dtype=jnp.float64))[None, ...]
jrb_mat.jrb_mat_hutchpp_trace_point(action_fn, probes, probes)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "mods": mods}))
"""
    )
    complex_hutchpp_wrapper_first_use = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import jcb_mat
def action_fn(v):
    return v
probes = jcb_mat._jcb_point_box(jnp.array([1.0+0.0j, 0.5+0.0j], dtype=jnp.complex128))[None, ...]
jcb_mat.jcb_mat_hutchpp_trace_point(action_fn, probes, probes)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "mods": mods}))
"""
    )

    assert operator_create["count"] <= import_tiers.MATRIX_FREE_OPERATOR_CREATE_FIRST_USE_MODULE_BUDGET
    assert operator_apply["count"] <= import_tiers.MATRIX_FREE_OPERATOR_APPLY_FIRST_USE_MODULE_BUDGET
    assert krylov_solve["count"] <= import_tiers.MATRIX_FREE_KRYLOV_SOLVE_FIRST_USE_MODULE_BUDGET
    assert implicit_adjoint["count"] <= import_tiers.MATRIX_FREE_IMPLICIT_ADJOINT_FIRST_USE_MODULE_BUDGET
    assert estimator_first_use["count"] <= import_tiers.MATRIX_FREE_ESTIMATOR_FIRST_USE_MODULE_BUDGET
    assert contour_first_use["count"] <= import_tiers.MATRIX_FREE_CONTOUR_FIRST_USE_MODULE_BUDGET
    assert real_slq_wrapper_first_use["count"] <= import_tiers.MATRIX_FREE_REAL_SLQ_WRAPPER_FIRST_USE_MODULE_BUDGET
    assert complex_slq_wrapper_first_use["count"] <= import_tiers.MATRIX_FREE_COMPLEX_SLQ_WRAPPER_FIRST_USE_MODULE_BUDGET
    assert real_hutchpp_wrapper_first_use["count"] <= import_tiers.MATRIX_FREE_REAL_HUTCHPP_WRAPPER_FIRST_USE_MODULE_BUDGET
    assert complex_hutchpp_wrapper_first_use["count"] <= import_tiers.MATRIX_FREE_COMPLEX_HUTCHPP_WRAPPER_FIRST_USE_MODULE_BUDGET

    assert "arbplusjax.matrix_free_core" in operator_create["mods"]
    assert "arbplusjax.matrix_free_estimators" not in operator_create["mods"]
    assert "arbplusjax.matrix_free_contour" not in operator_create["mods"]
    assert "arbplusjax.matrix_free_krylov" not in operator_create["mods"]
    assert "arbplusjax.matrix_free_adjoint" not in operator_create["mods"]
    assert "arbplusjax.iterative_solvers" not in operator_create["mods"]
    assert "arbplusjax.matfree_adjoints" not in operator_create["mods"]

    assert "arbplusjax.matrix_free_core" in operator_apply["mods"]
    assert "arbplusjax.matrix_free_estimators" not in operator_apply["mods"]
    assert "arbplusjax.matrix_free_contour" not in operator_apply["mods"]
    assert "arbplusjax.matrix_free_krylov" not in operator_apply["mods"]
    assert "arbplusjax.matrix_free_adjoint" not in operator_apply["mods"]
    assert "arbplusjax.iterative_solvers" not in operator_apply["mods"]
    assert "arbplusjax.matfree_adjoints" not in operator_apply["mods"]

    assert "arbplusjax.matrix_free_krylov" in krylov_solve["mods"]
    assert "arbplusjax.matrix_free_estimators" not in krylov_solve["mods"]
    assert "arbplusjax.matrix_free_contour" not in krylov_solve["mods"]
    assert "arbplusjax.iterative_solvers" in krylov_solve["mods"]
    assert "arbplusjax.matrix_free_adjoint" not in krylov_solve["mods"]
    assert "arbplusjax.matfree_adjoints" not in krylov_solve["mods"]

    assert "arbplusjax.matrix_free_adjoint" in implicit_adjoint["mods"]
    assert "arbplusjax.matrix_free_estimators" not in implicit_adjoint["mods"]
    assert "arbplusjax.matrix_free_contour" not in implicit_adjoint["mods"]
    assert "arbplusjax.iterative_solvers" in implicit_adjoint["mods"]
    assert "arbplusjax.matrix_free_krylov" not in implicit_adjoint["mods"]
    assert "arbplusjax.matfree_adjoints" not in implicit_adjoint["mods"]

    assert "arbplusjax.matrix_free_estimators" in estimator_first_use["mods"]
    assert "arbplusjax.matrix_free_contour" not in estimator_first_use["mods"]
    assert "arbplusjax.matrix_free_krylov" not in estimator_first_use["mods"]
    assert "arbplusjax.matrix_free_adjoint" not in estimator_first_use["mods"]
    assert "arbplusjax.iterative_solvers" not in estimator_first_use["mods"]
    assert "arbplusjax.matfree_adjoints" not in estimator_first_use["mods"]

    assert "arbplusjax.matrix_free_contour" in contour_first_use["mods"]
    assert "arbplusjax.matrix_free_estimators" not in contour_first_use["mods"]
    assert "arbplusjax.matrix_free_krylov" not in contour_first_use["mods"]
    assert "arbplusjax.matrix_free_adjoint" not in contour_first_use["mods"]
    assert "arbplusjax.iterative_solvers" not in contour_first_use["mods"]
    assert "arbplusjax.matfree_adjoints" not in contour_first_use["mods"]

    assert "arbplusjax.jrb_mat" in real_slq_wrapper_first_use["mods"]
    assert "arbplusjax.jrb_mat_slq_wrappers" in real_slq_wrapper_first_use["mods"]
    assert "arbplusjax.jrb_mat_contour_wrappers" not in real_slq_wrapper_first_use["mods"]
    assert "arbplusjax.matrix_free_contour" not in real_slq_wrapper_first_use["mods"]
    assert "arbplusjax.matrix_free_adjoint" not in real_slq_wrapper_first_use["mods"]
    assert "arbplusjax.matfree_adjoints" not in real_slq_wrapper_first_use["mods"]

    assert "arbplusjax.jcb_mat" in complex_slq_wrapper_first_use["mods"]
    assert "arbplusjax.jcb_mat_slq_wrappers" in complex_slq_wrapper_first_use["mods"]
    assert "arbplusjax.jcb_mat_contour_wrappers" not in complex_slq_wrapper_first_use["mods"]
    assert "arbplusjax.matrix_free_contour" not in complex_slq_wrapper_first_use["mods"]
    assert "arbplusjax.matrix_free_adjoint" not in complex_slq_wrapper_first_use["mods"]
    assert "arbplusjax.matfree_adjoints" not in complex_slq_wrapper_first_use["mods"]

    assert "arbplusjax.jrb_mat" in real_hutchpp_wrapper_first_use["mods"]
    assert "arbplusjax.jrb_mat_hutchpp_wrappers" in real_hutchpp_wrapper_first_use["mods"]
    assert "arbplusjax.jrb_mat_contour_wrappers" not in real_hutchpp_wrapper_first_use["mods"]
    assert "arbplusjax.matrix_free_contour" not in real_hutchpp_wrapper_first_use["mods"]
    assert "arbplusjax.matrix_free_adjoint" not in real_hutchpp_wrapper_first_use["mods"]
    assert "arbplusjax.matfree_adjoints" not in real_hutchpp_wrapper_first_use["mods"]

    assert "arbplusjax.jcb_mat" in complex_hutchpp_wrapper_first_use["mods"]
    assert "arbplusjax.jcb_mat_hutchpp_wrappers" in complex_hutchpp_wrapper_first_use["mods"]
    assert "arbplusjax.jcb_mat_contour_wrappers" not in complex_hutchpp_wrapper_first_use["mods"]
    assert "arbplusjax.matrix_free_contour" not in complex_hutchpp_wrapper_first_use["mods"]
    assert "arbplusjax.matrix_free_adjoint" not in complex_hutchpp_wrapper_first_use["mods"]
    assert "arbplusjax.matfree_adjoints" not in complex_hutchpp_wrapper_first_use["mods"]
