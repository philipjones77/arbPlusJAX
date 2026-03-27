from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from arbplusjax import import_tiers


REPO_ROOT = Path(__file__).resolve().parents[1]
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


def render() -> str:
    operator_create = _snapshot_operator_create()
    operator_apply = _snapshot_operator_apply()
    krylov_solve = _snapshot_krylov_solve()
    implicit_adjoint = _snapshot_implicit_adjoint_solve()
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
        "",
    ]
    sections = [
        ("Operator Creation (`dense_operator_plan(...)`)", operator_create, ["arbplusjax.matrix_free_core"], ["arbplusjax.matrix_free_krylov", "arbplusjax.matrix_free_adjoint", "arbplusjax.iterative_solvers", "arbplusjax.matfree_adjoints"]),
        ("Operator Apply (`operator_plan_apply(...)`)", operator_apply, ["arbplusjax.matrix_free_core"], ["arbplusjax.matrix_free_krylov", "arbplusjax.matrix_free_adjoint", "arbplusjax.iterative_solvers", "arbplusjax.matfree_adjoints"]),
        ("Krylov Solve (`krylov_solve_midpoint(...)`)", krylov_solve, ["arbplusjax.matrix_free_core", "arbplusjax.matrix_free_krylov", "arbplusjax.iterative_solvers"], ["arbplusjax.matrix_free_adjoint", "arbplusjax.matfree_adjoints"]),
        ("Implicit-Adjoint Solve (`implicit_krylov_solve_midpoint(...)`)", implicit_adjoint, ["arbplusjax.matrix_free_core", "arbplusjax.matrix_free_adjoint", "arbplusjax.iterative_solvers"], ["arbplusjax.matrix_free_krylov", "arbplusjax.matfree_adjoints"]),
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
            "- `matfree_adjoints` remains lazy and should load only when the explicit adjoint helper surface is selected",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_PATH.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
