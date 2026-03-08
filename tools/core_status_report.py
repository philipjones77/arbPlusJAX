from __future__ import annotations

import argparse
import ast
import os
import re
from dataclasses import dataclass
from pathlib import Path

from arbplusjax.function_provenance import engineering_status_for_public_name


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_WINDOWS_ROOT = Path(r"C:/Users/phili/OneDrive/Documents/GitHub")


def _resolve_root(env_var: str, default: Path, extra_candidates: tuple[Path, ...] = ()) -> Path:
    env = os.getenv(env_var, "").strip()
    if env:
        return Path(env)
    for cand in (default, *extra_candidates):
        if cand.exists():
            return cand
    return default


ARB_ROOT = _resolve_root(
    "ARB_ROOT",
    REPO_ROOT.parent / "arb",
    (LEGACY_WINDOWS_ROOT / "arb",),
)
JAX_ROOT = _resolve_root(
    "ARBPLUSJAX_ROOT",
    REPO_ROOT,
    (LEGACY_WINDOWS_ROOT / "arbPlusJAX",),
)
JAX_SRC = JAX_ROOT / "src" / "arbplusjax"


@dataclass(frozen=True)
class CoreFunctionStatus:
    name: str
    module: str
    implemented: bool
    basic: bool
    adaptive: bool
    rigorous_specialized: bool
    basic_only: bool
    kernel_split: str
    helper_consolidation: str
    batch: str
    ad: str
    hardening: str
    notes: str


FUNC_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _scan_c_prefix(prefix: str) -> set[str]:
    names: set[str] = set()
    if not ARB_ROOT.exists():
        return names
    for path in ARB_ROOT.glob("*.h"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if line.strip().startswith("#"):
                continue
            for match in FUNC_RE.finditer(line):
                name = match.group(1)
                if name.startswith(prefix):
                    names.add(name)
    return names


def _module_public_defs(path: Path, prefix: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith(prefix) and not node.name.startswith("_"):
            names.add(node.name)
    return names


def _adapter_name_sets(path: Path) -> tuple[set[str], set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    arb_rig: set[str] = set()
    arb_adapt: set[str] = set()
    acb_rig: set[str] = set()
    acb_adapt: set[str] = set()

    def collect_names(node: ast.FunctionDef) -> set[str]:
        names: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                value = child.value
                if value.startswith(("arb_", "acb_")) and value.endswith(("_prec", "_batch_prec")):
                    names.add(value)
        return names

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "_arb_rigorous_adapter":
            arb_rig |= collect_names(node)
        elif node.name == "_arb_adaptive_adapter":
            arb_adapt |= collect_names(node)
        elif node.name == "_acb_rigorous_adapter":
            acb_rig |= collect_names(node)
        elif node.name == "_acb_adaptive_adapter":
            acb_adapt |= collect_names(node)
    return arb_rig, arb_adapt, acb_rig, acb_adapt


def _normalize_core_base(name: str) -> str | None:
    if name.endswith("_batch_prec"):
        return name[:-11]
    if name.endswith("_prec"):
        return name[:-5]
    if name.endswith("_batch"):
        return name[:-6]
    return None


def _core_surface(module_path: Path, prefix: str) -> list[str]:
    public_defs = _module_public_defs(module_path, prefix)
    bases = sorted(
        name
        for name in public_defs
        if _normalize_core_base(name) is None and f"{name}_prec" in public_defs
    )
    return bases


def _status_rows() -> list[CoreFunctionStatus]:
    arb_core_path = JAX_SRC / "arb_core.py"
    acb_core_path = JAX_SRC / "acb_core.py"
    wrapper_path = JAX_SRC / "core_wrappers.py"

    arb_defs = _module_public_defs(arb_core_path, "arb_")
    acb_defs = _module_public_defs(acb_core_path, "acb_")
    arb_rig_prec, arb_adapt_prec, acb_rig_prec, acb_adapt_prec = _adapter_name_sets(wrapper_path)

    rows: list[CoreFunctionStatus] = []

    for name in _core_surface(arb_core_path, "arb_"):
        implemented = name in arb_defs
        basic = f"{name}_prec" in arb_defs
        adaptive = True
        rigorous_specialized = f"{name}_prec" in arb_rig_prec or f"{name}_batch_prec" in arb_rig_prec
        basic_only = implemented and basic and not adaptive and not rigorous_specialized
        eng = engineering_status_for_public_name(name) or {}
        if rigorous_specialized and adaptive:
            notes = "interval kernel is the rigorous path; adaptive uses dedicated or generic tightening"
        elif rigorous_specialized:
            notes = "interval kernel is the rigorous path"
        else:
            notes = "adaptive uses generic tightening"
        rows.append(
            CoreFunctionStatus(
                name=name,
                module="arb_core",
                implemented=implemented,
                basic=basic,
                adaptive=adaptive,
                rigorous_specialized=rigorous_specialized,
                basic_only=basic_only,
                kernel_split=eng.get("kernel_split", "shared_dispatch_separate_mode_kernels"),
                helper_consolidation=eng.get("helper_consolidation", "shared_elementary_or_core"),
                batch=eng.get("batch", "mixed"),
                ad=eng.get("ad", "mixed"),
                hardening=eng.get("hardening", "generic_or_mixed"),
                notes=notes,
            )
        )

    for name in _core_surface(acb_core_path, "acb_"):
        implemented = name in acb_defs
        basic = f"{name}_prec" in acb_defs
        rigorous_specialized = f"{name}_prec" in acb_rig_prec or f"{name}_batch_prec" in acb_rig_prec
        adaptive = (
            f"{name}_prec" in acb_adapt_prec
            or f"{name}_batch_prec" in acb_adapt_prec
            or (basic and not f"{name}_prec" in acb_adapt_prec and not f"{name}_batch_prec" in acb_adapt_prec)
        )
        basic_only = implemented and basic and not adaptive and not rigorous_specialized
        eng = engineering_status_for_public_name(name) or {}
        if rigorous_specialized and adaptive:
            notes = "specialized rigorous adapter and adaptive path present"
        elif rigorous_specialized:
            notes = "specialized rigorous adapter present; adaptive uses generic wrapper"
        elif adaptive:
            notes = "adaptive uses generic wrapper or dedicated adapter; rigorous uses generic wrapper"
        else:
            notes = "basic surface only"
        rows.append(
            CoreFunctionStatus(
                name=name,
                module="acb_core",
                implemented=implemented,
                basic=basic,
                adaptive=adaptive,
                rigorous_specialized=rigorous_specialized,
                basic_only=basic_only,
                kernel_split=eng.get("kernel_split", "shared_dispatch_separate_mode_kernels"),
                helper_consolidation=eng.get("helper_consolidation", "shared_elementary_or_core"),
                batch=eng.get("batch", "mixed"),
                ad=eng.get("ad", "mixed"),
                hardening=eng.get("hardening", "generic_or_mixed"),
                notes=notes,
            )
        )

    return rows


def _write_missing_report(prefix: str, module_name: str) -> None:
    c_names = _scan_c_prefix(prefix)
    if not c_names:
        raise RuntimeError(
            f"No C header functions found for prefix {prefix!r}. "
            f"Set ARB_ROOT to an Arb/FLINT source tree before regenerating missing reports."
        )
    module_defs = _module_public_defs(JAX_SRC / f"{module_name}.py", prefix)
    missing = sorted(name for name in c_names if name not in module_defs)
    out_path = JAX_ROOT / "docs" / "reports" / "missing_impls" / f"{module_name}_missing.txt"
    out_path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")


def _write_status_markdown(rows: list[CoreFunctionStatus]) -> None:
    out = JAX_ROOT / "docs" / "reports" / "core_function_status.md"
    total = len(rows)
    implemented = sum(1 for row in rows if row.implemented)
    adaptive = sum(1 for row in rows if row.adaptive)
    rigorous_specialized = sum(1 for row in rows if row.rigorous_specialized)
    basic_only = sum(1 for row in rows if row.basic_only)
    lines = [
        "Last updated: 2026-03-07T00:00:00Z",
        "",
        "# Core Function Status",
        "",
        "Generated from `src/arbplusjax/arb_core.py`, `src/arbplusjax/acb_core.py`, and `src/arbplusjax/core_wrappers.py`.",
        "",
        f"Summary: `implemented={implemented}/{total}`, `adaptive={adaptive}/{total}`, "
        f"`rigorous_specialized={rigorous_specialized}/{total}`, `basic_only={basic_only}/{total}`.",
        "",
        "Columns:",
        "- `implemented`: public function exists in the core module",
        "- `basic`: `*_prec` entry point exists",
        "- `adaptive`: adaptive mode path exists through a dedicated or generic wrapper",
        "- `rigorous_specialized`: function has a dedicated rigorous adapter in `core_wrappers.py`",
        "- `basic_only`: implemented/basic, but no adaptive path and no specialized rigorous path",
        "- `kernel_split`: whether the family shares dispatch while keeping separate mode kernels",
        "- `helper_consolidation`: whether constants/dtype/helper math has been centralized in shared substrate layers",
        "- `batch`, `ad`, `hardening`: current engineering state pulled from the repo-wide engineering policy registry",
        "",
        "| function | module | implemented | basic | adaptive | rigorous_specialized | basic_only | kernel_split | helper_consolidation | batch | ad | hardening | notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.name} | {row.module} | {'yes' if row.implemented else 'no'} | "
            f"{'yes' if row.basic else 'no'} | {'yes' if row.adaptive else 'no'} | "
            f"{'yes' if row.rigorous_specialized else 'no'} | {'yes' if row.basic_only else 'no'} | "
            f"{row.kernel_split} | {row.helper_consolidation} | {row.batch} | {row.ad} | {row.hardening} | {row.notes} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate core-function status and missing-implementation reports.")
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only write docs/reports/core_function_status.md. Skip header-based missing reports.",
    )
    args = parser.parse_args()

    _write_status_markdown(_status_rows())
    if not args.status_only:
        _write_missing_report("arb_", "arb_core")
        _write_missing_report("acb_", "acb_core")


if __name__ == "__main__":
    main()
