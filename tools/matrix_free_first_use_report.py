from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from arbplusjax import import_tiers

OUT_PATH = REPO_ROOT / "docs" / "reports" / "matrix_free_first_use_inventory.md"


def _run_python(code: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def _snapshot_operator_create() -> dict[str, object]:
    return _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import matrix_free_core as mfc
mfc.dense_operator_plan(jnp.eye(2, dtype=jnp.float32), orientation="forward", algebra="jrb")
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "modules": mods}))
"""
    )


def _snapshot_operator_apply() -> dict[str, object]:
    return _run_python(
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
print(json.dumps({"count": len(mods), "modules": mods}))
"""
    )


def _snapshot_krylov_solve() -> dict[str, object]:
    return _run_python(
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
print(json.dumps({"count": len(mods), "modules": mods}))
"""
    )


def _snapshot_implicit_adjoint_solve() -> dict[str, object]:
    return _run_python(
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
print(json.dumps({"count": len(mods), "modules": mods}))
"""
    )


def _snapshot_estimator_first_use() -> dict[str, object]:
    return _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import matrix_free_core as mfc
mfc.make_probe_estimate_statistics(jnp.array([1.0, 2.0], dtype=jnp.float32))
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "modules": mods}))
"""
    )


def _snapshot_contour_first_use() -> dict[str, object]:
    return _run_python(
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
print(json.dumps({"count": len(mods), "modules": mods}))
"""
    )


def _snapshot_real_slq_wrapper_first_use() -> dict[str, object]:
    return _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import jrb_mat
a = jrb_mat.jrb_mat_as_interval_matrix(jnp.array([[[2.0, 2.0], [0.0, 0.0]], [[0.0, 0.0], [3.0, 3.0]]], dtype=jnp.float64))
probes = jrb_mat.jrb_mat_as_interval_vector(jnp.array([[1.0, 1.0], [0.5, 0.5]], dtype=jnp.float64))[None, ...]
jrb_mat.jrb_mat_logdet_estimate_point(jrb_mat.jrb_mat_dense_operator(a), probes, steps=2)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "modules": mods}))
"""
    )


def _snapshot_complex_slq_wrapper_first_use() -> dict[str, object]:
    return _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import jcb_mat
a = jcb_mat._jcb_point_box(jnp.array([[2.0+0.0j, 0.0+0.0j], [0.0+0.0j, 3.0+0.0j]], dtype=jnp.complex128))
probes = jcb_mat._jcb_point_box(jnp.array([1.0+0.0j, 0.5+0.0j], dtype=jnp.complex128))[None, ...]
jcb_mat.jcb_mat_logdet_estimate_point(jcb_mat.jcb_mat_dense_operator(a), probes, steps=2)
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "modules": mods}))
"""
    )


def _snapshot_real_hutchpp_wrapper_first_use() -> dict[str, object]:
    return _run_python(
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
print(json.dumps({"count": len(mods), "modules": mods}))
"""
    )


def _snapshot_complex_hutchpp_wrapper_first_use() -> dict[str, object]:
    return _run_python(
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
print(json.dumps({"count": len(mods), "modules": mods}))
"""
    )


def render() -> str:
    operator_create = _snapshot_operator_create()
    operator_apply = _snapshot_operator_apply()
    krylov_solve = _snapshot_krylov_solve()
    implicit_adjoint = _snapshot_implicit_adjoint_solve()
    estimator_first_use = _snapshot_estimator_first_use()
    contour_first_use = _snapshot_contour_first_use()
    real_slq_wrapper_first_use = _snapshot_real_slq_wrapper_first_use()
    complex_slq_wrapper_first_use = _snapshot_complex_slq_wrapper_first_use()
    real_hutchpp_wrapper_first_use = _snapshot_real_hutchpp_wrapper_first_use()
    complex_hutchpp_wrapper_first_use = _snapshot_complex_hutchpp_wrapper_first_use()
    lines = [
        "Last updated: 2026-03-26T00:00:00Z",
        "",
        "# Matrix-Free First Use Inventory",
        "",
        "This report records the observed `arbplusjax.*` module set for representative matrix-free first-use boundaries.",
        "",
        "Budgets:",
        f"- operator creation budget: `<= {import_tiers.MATRIX_FREE_OPERATOR_CREATE_FIRST_USE_MODULE_BUDGET}`",
        f"- operator apply budget: `<= {import_tiers.MATRIX_FREE_OPERATOR_APPLY_FIRST_USE_MODULE_BUDGET}`",
        f"- Krylov solve budget: `<= {import_tiers.MATRIX_FREE_KRYLOV_SOLVE_FIRST_USE_MODULE_BUDGET}`",
        f"- implicit-adjoint solve budget: `<= {import_tiers.MATRIX_FREE_IMPLICIT_ADJOINT_FIRST_USE_MODULE_BUDGET}`",
        f"- estimator first use budget: `<= {import_tiers.MATRIX_FREE_ESTIMATOR_FIRST_USE_MODULE_BUDGET}`",
        f"- contour first use budget: `<= {import_tiers.MATRIX_FREE_CONTOUR_FIRST_USE_MODULE_BUDGET}`",
        f"- real SLQ wrapper first use budget: `<= {import_tiers.MATRIX_FREE_REAL_SLQ_WRAPPER_FIRST_USE_MODULE_BUDGET}`",
        f"- complex SLQ wrapper first use budget: `<= {import_tiers.MATRIX_FREE_COMPLEX_SLQ_WRAPPER_FIRST_USE_MODULE_BUDGET}`",
        f"- real Hutch++ wrapper first use budget: `<= {import_tiers.MATRIX_FREE_REAL_HUTCHPP_WRAPPER_FIRST_USE_MODULE_BUDGET}`",
        f"- complex Hutch++ wrapper first use budget: `<= {import_tiers.MATRIX_FREE_COMPLEX_HUTCHPP_WRAPPER_FIRST_USE_MODULE_BUDGET}`",
        "",
    ]
    sections = [
        ("Operator Creation (`dense_operator_plan(...)`)", operator_create, ["arbplusjax.matrix_free_core"], ["arbplusjax.matrix_free_estimators", "arbplusjax.matrix_free_contour", "arbplusjax.matrix_free_krylov", "arbplusjax.matrix_free_adjoint", "arbplusjax.iterative_solvers", "arbplusjax.matfree_adjoints"]),
        ("Operator Apply (`operator_plan_apply(...)`)", operator_apply, ["arbplusjax.matrix_free_core"], ["arbplusjax.matrix_free_estimators", "arbplusjax.matrix_free_contour", "arbplusjax.matrix_free_krylov", "arbplusjax.matrix_free_adjoint", "arbplusjax.iterative_solvers", "arbplusjax.matfree_adjoints"]),
        ("Krylov Solve (`krylov_solve_midpoint(...)`)", krylov_solve, ["arbplusjax.matrix_free_core", "arbplusjax.matrix_free_krylov", "arbplusjax.iterative_solvers"], ["arbplusjax.matrix_free_estimators", "arbplusjax.matrix_free_contour", "arbplusjax.matrix_free_adjoint", "arbplusjax.matfree_adjoints"]),
        ("Implicit-Adjoint Solve (`implicit_krylov_solve_midpoint(...)`)", implicit_adjoint, ["arbplusjax.matrix_free_core", "arbplusjax.matrix_free_adjoint", "arbplusjax.iterative_solvers"], ["arbplusjax.matrix_free_estimators", "arbplusjax.matrix_free_contour", "arbplusjax.matrix_free_krylov", "arbplusjax.matfree_adjoints"]),
        ("Estimator First Use (`make_probe_estimate_statistics(...)`)", estimator_first_use, ["arbplusjax.matrix_free_core", "arbplusjax.matrix_free_estimators"], ["arbplusjax.matrix_free_contour", "arbplusjax.matrix_free_krylov", "arbplusjax.matrix_free_adjoint", "arbplusjax.iterative_solvers", "arbplusjax.matfree_adjoints"]),
        ("Contour First Use (`contour_integral_action_point(...)`)", contour_first_use, ["arbplusjax.matrix_free_core", "arbplusjax.matrix_free_contour"], ["arbplusjax.matrix_free_estimators", "arbplusjax.matrix_free_krylov", "arbplusjax.matrix_free_adjoint", "arbplusjax.iterative_solvers", "arbplusjax.matfree_adjoints"]),
        ("Real SLQ Wrapper First Use (`jrb_mat_logdet_estimate_point(...)`)", real_slq_wrapper_first_use, ["arbplusjax.jrb_mat", "arbplusjax.jrb_mat_slq_wrappers"], ["arbplusjax.jrb_mat_contour_wrappers", "arbplusjax.matrix_free_contour", "arbplusjax.matrix_free_adjoint", "arbplusjax.matfree_adjoints"]),
        ("Complex SLQ Wrapper First Use (`jcb_mat_logdet_estimate_point(...)`)", complex_slq_wrapper_first_use, ["arbplusjax.jcb_mat", "arbplusjax.jcb_mat_slq_wrappers"], ["arbplusjax.jcb_mat_contour_wrappers", "arbplusjax.matrix_free_contour", "arbplusjax.matrix_free_adjoint", "arbplusjax.matfree_adjoints"]),
        ("Real Hutch++ Wrapper First Use (`jrb_mat_hutchpp_trace_point(...)`)", real_hutchpp_wrapper_first_use, ["arbplusjax.jrb_mat", "arbplusjax.jrb_mat_hutchpp_wrappers"], ["arbplusjax.jrb_mat_contour_wrappers", "arbplusjax.matrix_free_contour", "arbplusjax.matrix_free_adjoint", "arbplusjax.matfree_adjoints"]),
        ("Complex Hutch++ Wrapper First Use (`jcb_mat_hutchpp_trace_point(...)`)", complex_hutchpp_wrapper_first_use, ["arbplusjax.jcb_mat", "arbplusjax.jcb_mat_hutchpp_wrappers"], ["arbplusjax.jcb_mat_contour_wrappers", "arbplusjax.matrix_free_contour", "arbplusjax.matrix_free_adjoint", "arbplusjax.matfree_adjoints"]),
    ]
    for title, payload, expected_loaded, expected_unloaded in sections:
        lines.extend(
            [
                f"## {title}",
                "",
                f"- observed module count: `{payload['count']}`",
            ]
        )
        for mod in expected_loaded:
            lines.append(f"- `{mod}` loaded: `{mod in payload['modules']}`")
        for mod in expected_unloaded:
            lines.append(f"- `{mod}` loaded: `{mod in payload['modules']}`")
        lines.extend(
            [
                "",
                "```json",
                json.dumps(payload["modules"], indent=2),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            "- plain operator creation and primitive apply should stay on `matrix_free_core` only",
            "- Krylov solve should load the Krylov runtime layer without loading the implicit-adjoint runtime layer",
            "- implicit-adjoint solve should load the implicit-adjoint runtime layer without loading `matfree_adjoints` helper machinery",
            "- estimator helpers should load `matrix_free_estimators` and contour helpers should load `matrix_free_contour` without widening the operator-only path",
            "- real and complex SLQ wrapper first use should load only the selected wrapper module, not the contour or implicit-adjoint wrapper families",
            "- real and complex Hutch++ wrapper first use should load only the selected wrapper module, not the contour or implicit-adjoint wrapper families",
            "- `matfree_adjoints` remains lazy and should load only when the explicit adjoint helper surface is selected",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_PATH.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
