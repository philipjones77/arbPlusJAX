from __future__ import annotations

import argparse
import datetime as dt
import re
import zipfile
from pathlib import Path


ALLOWED_DIRS = {
    "src",
    "docs",
    "benchmarks",
    "examples",
    "tests",
    "tools",
}

ALLOWED_FILES = {
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "arbPlusJAX.code-workspace",
    "py.typed",
}

EXCLUDE_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "_bundles",
    "build",
    "dist",
    "results",
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dll",
    ".dylib",
    ".zip",
    ".whl",
    ".tar",
    ".gz",
    ".7z",
    ".exe",
    ".bin",
}

_SOURCE_NAME_RE = re.compile(r"^(?P<repo>[A-Za-z0-9._-]+)_source_(?P<date>\d{4}-\d{2}-\d{2})\.zip$")


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts.intersection(EXCLUDE_DIRS):
        return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    if path.name == ".env":
        return True
    return False


def is_valid_source_zip_name(filename: str, repo_name: str) -> bool:
    m = _SOURCE_NAME_RE.fullmatch(filename)
    if m is None or m.group("repo") != repo_name:
        return False
    try:
        dt.date.fromisoformat(m.group("date"))
    except ValueError:
        return False
    return True


def resolve_output_path(root: Path, output: str | None) -> Path:
    repo_name = root.name
    if output is None:
        stamp = dt.date.today().isoformat()
        out = (root / "_bundles" / f"{repo_name}_source_{stamp}.zip").resolve()
    else:
        out = (root / output).resolve()
    if not is_valid_source_zip_name(out.name, repo_name):
        raise ValueError(
            f"Output filename must match '{repo_name}_source_YYYY-MM-DD.zip'; got '{out.name}'."
        )
    return out


def build_zip(output_path: Path, root: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in ALLOWED_FILES:
            path = root / rel
            if path.is_file() and not should_skip(path):
                zf.write(path, rel)

        for rel_dir in ALLOWED_DIRS:
            base = root / rel_dir
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if path.is_dir():
                    continue
                if should_skip(path):
                    continue
                arcname = path.relative_to(root)
                zf.write(path, arcname.as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description="Package the arbPlusJAX repo into a zip file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output zip path relative to repo root.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_path = resolve_output_path(root, args.output)
    build_zip(output_path, root)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
