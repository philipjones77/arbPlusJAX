from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from arbplusjax import import_tiers
from tools import api_cold_path_report as acpr


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


def test_api_cold_path_report_is_current() -> None:
    path = REPO_ROOT / "docs" / "reports" / "api_cold_path_inventory.md"
    assert path.read_text(encoding="utf-8") == acpr.render()


def test_package_and_api_import_stay_within_module_budgets() -> None:
    package_payload = _run_python(
        """
import json
import sys
import arbplusjax
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "mods": mods}))
"""
    )
    api_payload = _run_python(
        """
import json
import sys
from arbplusjax import api
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "mods": mods}))
"""
    )

    assert package_payload["count"] <= import_tiers.PACKAGE_COLD_MODULE_BUDGET
    assert api_payload["count"] <= import_tiers.API_COLD_MODULE_BUDGET
    assert "arbplusjax.public_metadata" not in api_payload["mods"]
    assert "arbplusjax.point_wrappers_core" not in api_payload["mods"]
