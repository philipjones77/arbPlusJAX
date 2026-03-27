from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from arbplusjax import import_tiers


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "reports" / "api_cold_path_inventory.md"


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


def _package_snapshot() -> dict[str, object]:
    return _run_python(
        """
import json
import sys
import arbplusjax
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "modules": mods}))
"""
    )


def _api_snapshot() -> dict[str, object]:
    return _run_python(
        """
import json
import sys
from arbplusjax import api
mods = sorted(m for m in sys.modules if m.startswith("arbplusjax."))
print(json.dumps({"count": len(mods), "modules": mods}))
"""
    )


def render() -> str:
    package = _package_snapshot()
    api = _api_snapshot()
    package_modules = package["modules"]
    api_modules = api["modules"]
    lines = [
        "Last updated: 2026-03-26T00:00:00Z",
        "",
        "# API Cold Path Inventory",
        "",
        "This report records the observed module set for package import and `api` import after the current lazy-loading refactor.",
        "",
        "Budgets:",
        f"- package import budget: `<= {import_tiers.PACKAGE_COLD_MODULE_BUDGET}` `arbplusjax.*` modules",
        f"- `api` import budget: `<= {import_tiers.API_COLD_MODULE_BUDGET}` `arbplusjax.*` modules",
        "",
        "## Package Import",
        "",
        f"- observed module count: `{package['count']}`",
        "",
        "```json",
        json.dumps(package_modules, indent=2),
        "```",
        "",
        "## API Import",
        "",
        f"- observed module count: `{api['count']}`",
        f"- `public_metadata` loaded: `{'arbplusjax.public_metadata' in api_modules}`",
        f"- `point_wrappers_core` loaded: `{'arbplusjax.point_wrappers_core' in api_modules}`",
        "",
        "```json",
        json.dumps(api_modules, indent=2),
        "```",
        "",
        "## Notes",
        "",
        "- `point_wrappers_core` staying absent from this report means the point-family split is still holding.",
        "- `public_metadata` staying absent from this report means metadata access is no longer part of the `api` cold path.",
        "- Remaining cold-path bulk is currently dominated by the tail-acceleration runtime surface.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_PATH.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
