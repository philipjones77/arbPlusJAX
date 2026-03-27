from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from arbplusjax import import_tiers


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str) -> dict[str, bool] | list[str]:
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


def test_package_import_keeps_forbidden_tiers_unloaded() -> None:
    payload = _run_python(
        f"""
import json
import sys
import arbplusjax
forbidden = {sorted(import_tiers.API_COLD_FORBIDDEN)!r}
print(json.dumps([name for name in forbidden if name in sys.modules]))
"""
    )
    assert payload == []


def test_api_import_keeps_non_cold_tiers_unloaded() -> None:
    forbidden = sorted(
        import_tiers.POINT_ON_DEMAND_MODULES
        | import_tiers.INTERVAL_MODE_ON_DEMAND_MODULES
        | import_tiers.PROVIDER_BACKEND_ON_DEMAND_MODULES
        | import_tiers.BENCHMARK_DOCS_ONLY_MODULES
    )
    payload = _run_python(
        f"""
import json
import sys
from arbplusjax import api
forbidden = {forbidden!r}
print(json.dumps([name for name in forbidden if name in sys.modules]))
"""
    )
    assert payload == []


def test_import_tier_manifest_classifies_known_modules() -> None:
    assert import_tiers.classify_import_tier("arbplusjax") == "package_cold_allowed"
    assert import_tiers.classify_import_tier("arbplusjax.api") == "api_cold_allowed"
    assert import_tiers.classify_import_tier("arbplusjax.hypgeom") == "point_on_demand"
    assert import_tiers.classify_import_tier("arbplusjax.hypgeom_wrappers") == "interval_mode_on_demand"
    assert import_tiers.classify_import_tier("arbplusjax.mat_wrappers_dense") == "interval_mode_on_demand"
    assert import_tiers.classify_import_tier("arbplusjax.mat_wrappers_plans") == "interval_mode_on_demand"
    assert import_tiers.classify_import_tier("arbplusjax.boost_hypgeom") == "provider_backend_on_demand"
