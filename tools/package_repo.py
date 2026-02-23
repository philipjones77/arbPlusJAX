from __future__ import annotations

import argparse
import os
from pathlib import Path
import zipfile


EXCLUDE_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "results",
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dll",
    ".dylib",
}


def should_skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    parts = rel.parts
    if any(p in EXCLUDE_DIRS for p in parts):
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    return False


def build_zip(output_path: Path, root: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for dirpath, _, filenames in os.walk(root):
            dir_path = Path(dirpath)
            for name in filenames:
                file_path = dir_path / name
                if should_skip(file_path, root):
                    continue
                arcname = file_path.relative_to(root)
                zf.write(file_path, arcname.as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description="Package the arbPlusJAX repo into a zip file.")
    parser.add_argument(
        "--output",
        default="dist/arbPlusJAX.zip",
        help="Output zip path relative to repo root.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_path = (root / args.output).resolve()
    build_zip(output_path, root)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
