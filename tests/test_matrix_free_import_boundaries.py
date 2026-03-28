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
        assert "arbplusjax.matrix_free_contour" not in payload["mods"]
        assert "arbplusjax.matrix_free_estimators" not in payload["mods"]
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
    assert "arbplusjax.matrix_free_contour" not in krylov["mods"]
    assert "arbplusjax.matrix_free_estimators" not in krylov["mods"]
    assert "arbplusjax.matrix_free_adjoint" not in krylov["mods"]
    assert "arbplusjax.matfree_adjoints" not in krylov["mods"]

    assert "arbplusjax.matrix_free_adjoint" in implicit_adjoint["mods"]
    assert "arbplusjax.iterative_solvers" in implicit_adjoint["mods"]
    assert "arbplusjax.matrix_free_contour" not in implicit_adjoint["mods"]
    assert "arbplusjax.matrix_free_estimators" not in implicit_adjoint["mods"]
    assert "arbplusjax.matrix_free_krylov" not in implicit_adjoint["mods"]
    assert "arbplusjax.matfree_adjoints" not in implicit_adjoint["mods"]

    assert "arbplusjax.matfree_adjoints" in explicit_adjoints["mods"]


def test_jrb_mat_contour_wrappers_stay_lazy_until_selected() -> None:
    import_only = _run_python(
        """
import json
import sys
from arbplusjax import jrb_mat
del jrb_mat
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"mods": mods}))
"""
    )
    contour_use = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import jrb_mat
a = jrb_mat.jrb_mat_as_interval_matrix(jnp.array([[[4.0, 4.0], [0.0, 0.0]], [[0.0, 0.0], [9.0, 9.0]]], dtype=jnp.float64))
x = jrb_mat.jrb_mat_as_interval_vector(jnp.array([[1.0, 1.0], [2.0, 2.0]], dtype=jnp.float64))
_ = jrb_mat.jrb_mat_log_action_contour_point(
    jrb_mat.jrb_mat_dense_operator(a),
    x,
    center=6.5 + 0.0j,
    radius=3.0,
    quadrature_order=8,
)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"mods": mods}))
"""
    )

    assert "arbplusjax.jrb_mat_contour_wrappers" not in import_only["mods"]
    assert "arbplusjax.jrb_mat_contour_wrappers" in contour_use["mods"]


def test_jcb_mat_contour_wrappers_stay_lazy_until_selected() -> None:
    import_only = _run_python(
        """
import json
import sys
from arbplusjax import jcb_mat
del jcb_mat
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"mods": mods}))
"""
    )
    contour_use = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import jcb_mat
a = jcb_mat._jcb_point_box(jnp.array([[2.0+0.0j, 0.0+0.0j], [0.0+0.0j, 3.0+0.0j]], dtype=jnp.complex128))
x = jcb_mat._jcb_point_box(jnp.array([1.0+0.0j, 2.0+0.0j], dtype=jnp.complex128))
_ = jcb_mat.jcb_mat_log_action_contour_point(
    jcb_mat.jcb_mat_dense_operator(a),
    x,
    center=2.5 + 0.0j,
    radius=0.9,
    quadrature_order=8,
)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"mods": mods}))
"""
    )

    assert "arbplusjax.jcb_mat_contour_wrappers" not in import_only["mods"]
    assert "arbplusjax.jcb_mat_contour_wrappers" in contour_use["mods"]


def test_jrb_mat_slq_wrappers_stay_lazy_until_selected() -> None:
    import_only = _run_python(
        """
import json
import sys
from arbplusjax import jrb_mat
del jrb_mat
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"mods": mods}))
"""
    )
    slq_use = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import jrb_mat
a = jrb_mat.jrb_mat_as_interval_matrix(jnp.array([[[2.0, 2.0], [0.0, 0.0]], [[0.0, 0.0], [3.0, 3.0]]], dtype=jnp.float64))
probes = jrb_mat.jrb_mat_as_interval_vector(jnp.array([[1.0, 1.0], [0.5, 0.5]], dtype=jnp.float64))[None, ...]
_ = jrb_mat.jrb_mat_logdet_estimate_point(jrb_mat.jrb_mat_dense_operator(a), probes, steps=2)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"mods": mods}))
"""
    )

    assert "arbplusjax.jrb_mat_slq_wrappers" not in import_only["mods"]
    assert "arbplusjax.jrb_mat_slq_wrappers" in slq_use["mods"]


def test_jcb_mat_slq_wrappers_stay_lazy_until_selected() -> None:
    import_only = _run_python(
        """
import json
import sys
from arbplusjax import jcb_mat
del jcb_mat
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"mods": mods}))
"""
    )
    slq_use = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import jcb_mat
a = jcb_mat._jcb_point_box(jnp.array([[2.0+0.0j, 0.0+0.0j], [0.0+0.0j, 3.0+0.0j]], dtype=jnp.complex128))
probes = jcb_mat._jcb_point_box(jnp.array([1.0+0.0j, 0.5+0.0j], dtype=jnp.complex128))[None, ...]
_ = jcb_mat.jcb_mat_logdet_estimate_point(jcb_mat.jcb_mat_dense_operator(a), probes, steps=2)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"mods": mods}))
"""
    )

    assert "arbplusjax.jcb_mat_slq_wrappers" not in import_only["mods"]
    assert "arbplusjax.jcb_mat_slq_wrappers" in slq_use["mods"]


def test_jrb_mat_hutchpp_wrappers_stay_lazy_until_selected() -> None:
    import_only = _run_python(
        """
import json
import sys
from arbplusjax import jrb_mat
del jrb_mat
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"mods": mods}))
"""
    )
    hutchpp_use = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import jrb_mat
def action_fn(v):
    return v
probes = jrb_mat.jrb_mat_as_interval_vector(jnp.array([[1.0, 1.0], [0.5, 0.5]], dtype=jnp.float64))[None, ...]
_ = jrb_mat.jrb_mat_hutchpp_trace_point(action_fn, probes, probes)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"mods": mods}))
"""
    )

    assert "arbplusjax.jrb_mat_hutchpp_wrappers" not in import_only["mods"]
    assert "arbplusjax.jrb_mat_hutchpp_wrappers" in hutchpp_use["mods"]


def test_jcb_mat_hutchpp_wrappers_stay_lazy_until_selected() -> None:
    import_only = _run_python(
        """
import json
import sys
from arbplusjax import jcb_mat
del jcb_mat
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"mods": mods}))
"""
    )
    hutchpp_use = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import jcb_mat
def action_fn(v):
    return v
probes = jcb_mat._jcb_point_box(jnp.array([1.0+0.0j, 0.5+0.0j], dtype=jnp.complex128))[None, ...]
_ = jcb_mat.jcb_mat_hutchpp_trace_point(action_fn, probes, probes)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"mods": mods}))
"""
    )

    assert "arbplusjax.jcb_mat_hutchpp_wrappers" not in import_only["mods"]
    assert "arbplusjax.jcb_mat_hutchpp_wrappers" in hutchpp_use["mods"]


def test_jrb_mat_leja_hutchpp_wrappers_stay_lazy_until_selected() -> None:
    import_only = _run_python(
        """
import json
import sys
from arbplusjax import jrb_mat
del jrb_mat
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"mods": mods}))
"""
    )
    leja_use = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import jrb_mat
a = jrb_mat.jrb_mat_as_interval_matrix(jnp.array([[[2.0, 2.0], [0.0, 0.0]], [[0.0, 0.0], [3.0, 3.0]]], dtype=jnp.float64))
probes = jrb_mat.jrb_mat_as_interval_vector(jnp.array([[1.0, 1.0], [0.5, 0.5]], dtype=jnp.float64))[None, ...]
_ = jrb_mat.jrb_mat_logdet_leja_hutchpp_point(
    jrb_mat.jrb_mat_dense_operator(a),
    probes,
    jnp.zeros((0, 2, 2), dtype=jnp.float64),
    degree=6,
    spectral_bounds=(2.0, 3.0),
)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"mods": mods}))
"""
    )

    assert "arbplusjax.jrb_mat_hutchpp_wrappers" not in import_only["mods"]
    assert "arbplusjax.jrb_mat_hutchpp_wrappers" in leja_use["mods"]


def test_jcb_mat_leja_hutchpp_wrappers_stay_lazy_until_selected() -> None:
    import_only = _run_python(
        """
import json
import sys
from arbplusjax import jcb_mat
del jcb_mat
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"mods": mods}))
"""
    )
    leja_use = _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import jcb_mat
a = jcb_mat._jcb_point_box(jnp.array([[2.0+0.0j, 0.0+0.0j], [0.0+0.0j, 3.0+0.0j]], dtype=jnp.complex128))
probes = jcb_mat._jcb_point_box(jnp.array([1.0+0.0j, 0.5+0.0j], dtype=jnp.complex128))[None, ...]
_ = jcb_mat.jcb_mat_logdet_leja_hutchpp_point(
    jcb_mat.jcb_mat_dense_operator(a),
    probes,
    jnp.zeros((0, 2, 4), dtype=jnp.float64),
    degree=6,
    spectral_bounds=(2.0, 3.0),
)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"mods": mods}))
"""
    )

    assert "arbplusjax.jcb_mat_hutchpp_wrappers" not in import_only["mods"]
    assert "arbplusjax.jcb_mat_hutchpp_wrappers" in leja_use["mods"]
