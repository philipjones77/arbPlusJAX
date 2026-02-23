from __future__ import annotations

import ast
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

ARB_ROOT = Path(r"C:/Users/phili/OneDrive/Documents/GitHub/arb")
JAX_ROOT = Path(r"C:/Users/phili/OneDrive/Documents/GitHub/arbPlusJAX")
JAX_SRC = JAX_ROOT / "src" / "arbplusjax"
TESTS_ROOT = JAX_ROOT / "tests"
MPMATH_ROOT = Path(r"C:/Users/phili/OneDrive/Documents/GitHub/mpmath")

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


def _scan_jax_scipy_special() -> Set[str]:
    names: Set[str] = set()
    try:
        import jax.scipy.special as jsp  # type: ignore

        for name in dir(jsp):
            if name.startswith("_"):
                continue
            attr = getattr(jsp, name)
            if callable(attr):
                names.add(name)
    except Exception:
        pass
    return names


def main() -> None:
    c_names = _scan_c_headers()
    jax_defs = _scan_jax_defs()
    tests_text = _scan_tests_text()
    mpmath_names = _scan_mpmath_names()
    jax_scipy_names = _scan_jax_scipy_special()

    jax_funcs = set(jax_defs.keys())
    c_missing = sorted(n for n in c_names if n not in jax_funcs)
    c_implemented = sorted(n for n in c_names if n in jax_funcs)

    tested = set()
    for name in jax_funcs:
        if re.search(rf"\b{name}\b", tests_text):
            tested.add(name)

    untested = sorted(n for n in jax_funcs if n not in tested)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
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

        f.write("\n## jax.scipy.special overlap\n\n")
        overlap_jsp = sorted(n for n in jax_funcs if n in jax_scipy_names)
        for name in overlap_jsp[:200]:
            f.write(f"- {name}\n")
        if len(overlap_jsp) > 200:
            f.write(f"- ... ({len(overlap_jsp)} total)\n")

    # targets registry
    targets_csv = TESTS_ROOT / "targets.csv"
    with targets_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["function", "module", "path", "in_c", "tested", "in_mpmath", "in_jax_scipy"])
        for name in sorted(jax_funcs):
            jd = jax_defs[name]
            writer.writerow([
                name,
                jd.module,
                jd.path,
                "yes" if name in c_names else "no",
                "yes" if name in tested else "no",
                "yes" if name in mpmath_names else "no",
                "yes" if name in jax_scipy_names else "no",
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

    todo = JAX_ROOT / "docs" / "todo.md"
    with todo.open("w", encoding="utf-8") as f:
        f.write("# TODO\n\n")
        f.write("## Missing C implementations\n\n")
        for name in c_missing:
            f.write(f"- {name}\n")
        f.write("\n## Untested JAX functions\n\n")
        for name in untested:
            f.write(f"- {name}\n")


if __name__ == "__main__":
    main()
