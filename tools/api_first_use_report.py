from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from arbplusjax import import_tiers


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "reports" / "api_first_use_inventory.md"


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


def _snapshot_api_import() -> dict[str, object]:
    return _run_python(
        """
import json
import sys
from arbplusjax import api
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "modules": mods}))
"""
    )


def _snapshot_point_core() -> dict[str, object]:
    return _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import api
api.eval_point("exp", jnp.array([0.5], dtype=jnp.float32), dtype="float32")
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "modules": mods}))
"""
    )


def _snapshot_point_matrix() -> dict[str, object]:
    return _run_python(
        """
import json
import sys
from arbplusjax import api
api.eval_point("arb_mat_zero", 2, dtype="float32")
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "modules": mods}))
"""
    )


def _snapshot_point_matrix_plan_prepare() -> dict[str, object]:
    return _run_python(
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
print(json.dumps({"count": len(mods), "modules": mods}))
"""
    )


def _snapshot_point_matrix_plan_apply() -> dict[str, object]:
    return _run_python(
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
print(json.dumps({"count": len(mods), "modules": mods}))
"""
    )


def _snapshot_tail() -> dict[str, object]:
    return _run_python(
        """
import json
import sys
import jax.numpy as jnp
from arbplusjax import api
api.tail_integral(lambda t: jnp.exp(-t), jnp.array(1.0, dtype=jnp.float32))
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "modules": mods}))
"""
    )


def render() -> str:
    api_import = _snapshot_api_import()
    point_core = _snapshot_point_core()
    point_matrix = _snapshot_point_matrix()
    point_matrix_plan_prepare = _snapshot_point_matrix_plan_prepare()
    point_matrix_plan_apply = _snapshot_point_matrix_plan_apply()
    tail = _snapshot_tail()
    lines = [
        "Last updated: 2026-03-26T00:00:00Z",
        "",
        "# API First Use Inventory",
        "",
        "This report records the observed `arbplusjax.*` module set after representative first-use calls on top of `from arbplusjax import api`.",
        "",
        "Budgets:",
        f"- `api` import budget: `<= {import_tiers.API_COLD_MODULE_BUDGET}`",
        f"- core point first-use budget: `<= {import_tiers.POINT_CORE_FIRST_USE_MODULE_BUDGET}`",
        f"- matrix dense first-use budget: `<= {import_tiers.POINT_MATRIX_DENSE_FIRST_USE_MODULE_BUDGET}`",
        f"- matrix plan-prepare first-use budget: `<= {import_tiers.POINT_MATRIX_PLAN_PREPARE_FIRST_USE_MODULE_BUDGET}`",
        f"- matrix plan-apply first-use budget: `<= {import_tiers.POINT_MATRIX_PLAN_APPLY_FIRST_USE_MODULE_BUDGET}`",
        f"- tail first-use budget: `<= {import_tiers.TAIL_FIRST_USE_MODULE_BUDGET}`",
        "",
    ]
    sections = [
        ("API Import", api_import, []),
        ("Core Point First Use (`eval_point(\"exp\", ...)`)", point_core, ["arbplusjax.point_wrappers_core"]),
        ("Matrix Dense First Use (`eval_point(\"arb_mat_zero\", ...)`)", point_matrix, ["arbplusjax.point_wrappers_matrix_dense", "arbplusjax.mat_common"]),
        ("Matrix Plan Prepare First Use (`eval_point(\"arb_mat_matvec_cached_prepare\", ...)`)", point_matrix_plan_prepare, ["arbplusjax.point_wrappers_matrix_plans", "arbplusjax.mat_common"]),
        ("Matrix Plan Apply First Use (`eval_point(\"arb_mat_matvec_cached_apply\", ...)`)", point_matrix_plan_apply, ["arbplusjax.point_wrappers_matrix_plans", "arbplusjax.mat_common"]),
        ("Tail First Use (`tail_integral(...)`)", tail, ["arbplusjax.special.tail_acceleration"]),
    ]
    for title, payload, highlights in sections:
        lines.extend(
            [
                f"## {title}",
                "",
                f"- observed module count: `{payload['count']}`",
            ]
        )
        for mod in highlights:
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
            "- The core point path should load `point_wrappers_core` and nothing matrix-specific.",
            "- The matrix dense point path should load `point_wrappers_matrix_dense` plus `mat_common` and remain off plan-only helpers.",
            "- The matrix plan paths should load `point_wrappers_matrix_plans` and stay off interval wrappers.",
            "- The tail path is intentionally larger because it brings in the tail-acceleration subsystem only on demand.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_PATH.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
