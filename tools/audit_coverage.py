from __future__ import annotations

import ast
import csv
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_WINDOWS_ROOT = Path(r"C:/Users/phili/OneDrive/Documents/GitHub")


def _resolve_root(env_var: str, default: Path, extra_candidates: Tuple[Path, ...] = ()) -> Path:
    env = os.getenv(env_var, "").strip()
    if env:
        return Path(env)
    for cand in (default, *extra_candidates):
        if cand.exists():
            return cand
    return default


ARB_ROOT = _resolve_root(
    "ARB_ROOT",
    _REPO_ROOT.parent / "arb",
    (_LEGACY_WINDOWS_ROOT / "arb",),
)
JAX_ROOT = _resolve_root(
    "ARBPLUSJAX_ROOT",
    _REPO_ROOT,
    (_LEGACY_WINDOWS_ROOT / "arbPlusJAX",),
)
JAX_SRC = JAX_ROOT / "src" / "arbplusjax"
TESTS_ROOT = JAX_ROOT / "tests"
MPMATH_ROOT = _resolve_root(
    "MPMATH_ROOT",
    _REPO_ROOT.parent / "mpmath",
    (_LEGACY_WINDOWS_ROOT / "mpmath",),
)

PREFIXES = (
    "arb_",
    "acb_",
    "arf_",
    "fmpr_",
    "fmpz_",
    "fmpzi_",
    "mag_",
    "acf_",
    "bool_mat_",
    "arb_poly_",
    "acb_poly_",
    "arb_mat_",
    "acb_mat_",
    "arb_calc_",
    "acb_calc_",
    "arb_hypgeom_",
    "acb_hypgeom_",
    "arb_fpwrap_",
    "dft_",
    "acb_dft_",
    "dirichlet_",
    "acb_dirichlet_",
    "acb_modular_",
    "acb_elliptic_",
)

EXCLUDE_NAMES = {
    "main",
}


@dataclass
class JaxDef:
    name: str
    module: str
    path: str


FUNC_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _scan_c_headers() -> Set[str]:
    names: Set[str] = set()
    for path in ARB_ROOT.glob("*.h"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if line.strip().startswith("#"):
                continue
            for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", line):
                name = m.group(1)
                if name in EXCLUDE_NAMES:
                    continue
                if name.startswith(PREFIXES):
                    names.add(name)
    return names


def _git_show_text(spec: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "show", spec],
            cwd=str(JAX_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return ""
    return completed.stdout


def _parse_missing_from_audit_text(text: str) -> Set[str]:
    names: Set[str] = set()
    in_missing = False
    for line in text.splitlines():
        if line.strip() == "## Missing C functions in JAX":
            in_missing = True
            continue
        if in_missing and line.startswith("## "):
            break
        if in_missing and line.startswith("- "):
            names.add(line[2:].strip())
    return names


def _parse_in_c_from_targets_text(text: str) -> Set[str]:
    names: Set[str] = set()
    if not text.strip():
        return names
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        if row.get("in_c", "").strip().lower() == "yes":
            fn = row.get("function", "").strip()
            if fn:
                names.add(fn)
    return names


def _fallback_c_inventory_from_git() -> Set[str]:
    audit_text = _git_show_text("HEAD~1:docs/audit.md")
    targets_text = _git_show_text("HEAD~1:tests/targets.csv")
    return _parse_missing_from_audit_text(audit_text) | _parse_in_c_from_targets_text(targets_text)


def _scan_jax_defs() -> Dict[str, JaxDef]:
    defs: Dict[str, JaxDef] = {}
    for path in JAX_SRC.glob("*.py"):
        if path.name.startswith("_"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                name = node.name
                if name.startswith("_"):
                    continue
                defs[name] = JaxDef(name=name, module=path.stem, path=str(path))
    return defs


def _scan_tests_text() -> str:
    buf = []
    for path in TESTS_ROOT.rglob("*.py"):
        buf.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(buf)


def _scan_mpmath_names() -> Set[str]:
    names: Set[str] = set()
    init = MPMATH_ROOT / "mpmath" / "__init__.py"
    if not init.exists():
        return names
    text = init.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if line.startswith("def "):
            name = line.split()[1].split("(")[0]
            names.add(name)
        if line.startswith("from .") and "import" in line:
            parts = line.split("import", 1)[1]
            for token in parts.split(","):
                nm = token.strip().split(" as ")[0].strip()
                if nm:
                    names.add(nm)
    return names


def _scan_jax_scipy_names() -> Set[str]:
    names: Set[str] = set()
    try:
        import jax.scipy as jsp  # type: ignore
    except Exception:
        return names

    for attr in ("special", "linalg", "stats", "fft"):
        mod = getattr(jsp, attr, None)
        if mod is None:
            continue
        for name in dir(mod):
            if not name.startswith("_"):
                names.add(name)
    return names


def main() -> None:
    c_names = _scan_c_headers()
    if not c_names:
        c_names = _fallback_c_inventory_from_git()
    jax_defs = _scan_jax_defs()
    tests_text = _scan_tests_text()
    mpmath_names = _scan_mpmath_names()
    jax_scipy_names = _scan_jax_scipy_names()

    jax_funcs = set(jax_defs.keys())
    c_missing = sorted(n for n in c_names if n not in jax_funcs)
    c_implemented = sorted(n for n in c_names if n in jax_funcs)

    tested = set()
    for name in jax_funcs:
        if re.search(rf"\b{name}\b", tests_text):
            tested.add(name)

    untested = sorted(n for n in jax_funcs if n not in tested)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    results_dir = JAX_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    audit_md = results_dir / f"audit_{timestamp}.md"

    with audit_md.open("w", encoding="utf-8") as f:
        f.write(f"# Audit Summary ({timestamp})\n\n")
        f.write(f"C headers scanned: {len(c_names)} functions\n\n")
        f.write(f"JAX functions (public defs): {len(jax_funcs)}\n\n")
        f.write(f"C functions implemented in JAX: {len(c_implemented)}\n\n")
        f.write(f"C functions missing in JAX: {len(c_missing)}\n\n")
        f.write(f"Tested JAX functions: {len(tested)}\n\n")
        f.write(f"Untested JAX functions: {len(untested)}\n\n")

        f.write("## Missing C functions in JAX\n\n")
        for name in c_missing:
            f.write(f"- {name}\n")

        f.write("\n## Untested JAX functions\n\n")
        for name in untested:
            f.write(f"- {name}\n")

        f.write("\n## Mpmath overlap\n\n")
        overlap_mpmath = sorted(n for n in jax_funcs if n in mpmath_names)
        for name in overlap_mpmath[:200]:
            f.write(f"- {name}\n")
        if len(overlap_mpmath) > 200:
            f.write(f"- ... ({len(overlap_mpmath)} total)\n")

    # targets registry
    targets_csv = TESTS_ROOT / "targets.csv"
    with targets_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["function", "module", "path", "in_c", "tested", "in_mpmath"])
        for name in sorted(jax_funcs):
            jd = jax_defs[name]
            writer.writerow([
                name,
                jd.module,
                jd.path,
                "yes" if name in c_names else "no",
                "yes" if name in tested else "no",
                "yes" if name in mpmath_names else "no",
            ])

    targets_md = TESTS_ROOT / "targets.md"
    with targets_md.open("w", encoding="utf-8") as f:
        f.write(f"# Test Targets ({timestamp})\n\n")
        f.write(f"Total functions: {len(jax_funcs)}\n\n")
        f.write("| function | module | in_c | tested | in_mpmath | in_jax_scipy |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name in sorted(jax_funcs):
            jd = jax_defs[name]
            f.write(
                f"| {name} | {jd.module} | {'yes' if name in c_names else 'no'} | {'yes' if name in tested else 'no'} | {'yes' if name in mpmath_names else 'no'} | {'yes' if name in jax_scipy_names else 'no'} |\n"
            )

    # docs
    docs_audit = JAX_ROOT / "docs" / "audit.md"
    docs_audit.write_text(audit_md.read_text(encoding="utf-8"), encoding="utf-8")

if __name__ == "__main__":
    main()
