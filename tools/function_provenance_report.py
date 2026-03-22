from __future__ import annotations

import argparse
from pathlib import Path

from arbplusjax import capability_registry as cr
from arbplusjax import function_provenance as fp


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate function provenance and naming reports.")
    parser.add_argument("--lookup", help="Show all registered implementations for a base function name.")
    args = parser.parse_args()
    if args.lookup:
        print(fp.render_lookup(args.lookup), end="")
        return
    _write(REPO_ROOT / "docs" / "reports" / "function_provenance_registry.md", fp.render_registry_summary())
    _write(REPO_ROOT / "docs" / "reports" / "function_capability_registry.json", cr.render_capability_registry_json())
    _write(REPO_ROOT / "docs" / "reports" / "function_implementation_index.md", fp.render_implementation_index())
    _write(REPO_ROOT / "docs" / "reports" / "function_engineering_status.md", fp.render_engineering_status())
    _write(REPO_ROOT / "docs" / "reports" / "arb_like_functions.md", fp.render_report("arb_like", "Arb-like Functions"))
    _write(REPO_ROOT / "docs" / "reports" / "alternative_functions.md", fp.render_report("alternative", "Alternative Functions"))
    _write(REPO_ROOT / "docs" / "reports" / "new_functions.md", fp.render_report("new", "New Functions"))


if __name__ == "__main__":
    main()
