from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prepare_sphinx_docs import DEFAULT_OUTPUT_ROOT, DEFAULT_SOURCE_ROOT, prepare_docs


PYTHON = sys.executable
DOCS_ROOT = REPO_ROOT / "docs"


def build_docs(builder: str, *, source_root: Path, build_root: Path) -> None:
    prepare_docs(source_root)
    if builder == "latexpdf":
        cmd = [
            "sphinx-build",
            "-c",
            str(DOCS_ROOT),
            "-M",
            "latexpdf",
            str(source_root),
            str(build_root),
        ]
    else:
        out_dir = build_root / builder
        cmd = [
            "sphinx-build",
            "-c",
            str(DOCS_ROOT),
            "-b",
            builder,
            str(source_root),
            str(out_dir),
        ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Sphinx/MyST docs site from the Markdown docs tree.")
    parser.add_argument(
        "--builder",
        choices=["html", "linkcheck", "latexpdf"],
        default="html",
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--build-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "_build")
    args = parser.parse_args()
    build_docs(
        args.builder,
        source_root=args.source_root.resolve(),
        build_root=args.build_root.resolve(),
    )


if __name__ == "__main__":
    main()
