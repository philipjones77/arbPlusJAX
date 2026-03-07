from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "arbplusjax"


@dataclass(frozen=True)
class PointStatus:
    name: str
    module: str
    point_wrapper: str
    available: bool


def _public_defs(path: Path, prefix: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith(prefix) and not node.name.startswith("_"):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith(prefix) and not target.id.startswith("_"):
                    out.add(target.id)
    return out


def _core_surface(path: Path, prefix: str) -> list[str]:
    defs = _public_defs(path, prefix)
    return sorted(
        name
        for name in defs
        if f"{name}_prec" in defs and not name.endswith("_batch")
    )


def _rows() -> list[PointStatus]:
    point_defs = _public_defs(SRC / "point_wrappers.py", "")
    rows: list[PointStatus] = []
    for module_name, prefix in (("arb_core", "arb_"), ("acb_core", "acb_")):
        for name in _core_surface(SRC / f"{module_name}.py", prefix):
            wrapper = f"{name}_point"
            rows.append(PointStatus(name=name, module=module_name, point_wrapper=wrapper, available=wrapper in point_defs))
    return rows


def main() -> None:
    rows = _rows()
    total = len(rows)
    available = sum(1 for row in rows if row.available)
    lines = [
        "Last updated: 2026-03-07T00:00:00Z",
        "",
        "# Core Point Status",
        "",
        "Generated from `arb_core.py`, `acb_core.py`, and `point_wrappers.py`.",
        "",
        f"Summary: `point_available={available}/{total}`.",
        "",
        "| function | module | point_wrapper | available |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row.name} | {row.module} | {row.point_wrapper} | {'yes' if row.available else 'no'} |")
    (REPO_ROOT / "docs" / "reports" / "core_point_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
