from __future__ import annotations

import argparse
import os
import posixpath
import re
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "docs_site"
DEFAULT_SOURCE_ROOT = DEFAULT_OUTPUT_ROOT / "source"

MARKDOWN_SUFFIXES = {".md", ".ipynb"}
REPO_DOC_LINK_RE = re.compile(r"(\[[^\]]+\])\((/[^)]+)\)")
HTML_ATTR_LINK_RE = re.compile(r'((?:href|src))="(/[^"]+)"')
SITE_COPY_FOLDERS = [
    "governance",
    "implementation",
    "notation",
    "objects",
    "practical",
    "reports",
    "specs",
    "standards",
    "status",
    "theory",
]


def _git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def _github_base_url() -> str:
    env_repo = os.environ.get("GITHUB_REPOSITORY")
    if env_repo:
        return f"https://github.com/{env_repo}"

    remote = _git_output(["git", "config", "--get", "remote.origin.url"])
    if not remote:
        return "https://github.com/philipturner/arbplusJAX"
    if remote.startswith("git@github.com:"):
        repo = remote.split(":", 1)[1]
    elif "github.com/" in remote:
        repo = remote.split("github.com/", 1)[1]
    else:
        return "https://github.com/philipturner/arbplusJAX"
    repo = repo.removesuffix(".git")
    return f"https://github.com/{repo}"


def _github_ref() -> str:
    return (
        os.environ.get("GITHUB_REF_NAME")
        or _git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        or "main"
    )


def _split_anchor(target: str) -> tuple[str, str]:
    if "#" not in target:
        return target, ""
    path, anchor = target.split("#", 1)
    return path, f"#{anchor}"


def _relative_doc_link(current_path: Path, target_path: str, source_root: Path) -> str:
    current_rel = current_path.relative_to(source_root)
    target_rel = Path(target_path)
    rel = posixpath.relpath(target_rel.as_posix(), start=current_rel.parent.as_posix())
    return rel


def _rewrite_repo_root_link(
    current_path: Path,
    target: str,
    github_base: str,
    github_ref: str,
    source_root: Path,
) -> str:
    path_part, anchor = _split_anchor(target)
    if path_part.startswith("/docs/"):
        doc_target = path_part[len("/docs/") :]
        return f"{_relative_doc_link(current_path, doc_target, source_root)}{anchor}"
    repo_rel = path_part.lstrip("/")
    return f"{github_base}/blob/{github_ref}/{repo_rel}{anchor}"


def _rewrite_markdown_links(
    text: str,
    current_path: Path,
    github_base: str,
    github_ref: str,
    source_root: Path,
) -> str:
    def replace_markdown(match: re.Match[str]) -> str:
        label, target = match.groups()
        if not target.startswith("/"):
            return match.group(0)
        return f"{label}({_rewrite_repo_root_link(current_path, target, github_base, github_ref, source_root)})"

    def replace_html(match: re.Match[str]) -> str:
        attr, target = match.groups()
        if not target.startswith("/"):
            return match.group(0)
        rewritten = _rewrite_repo_root_link(current_path, target, github_base, github_ref, source_root)
        return f'{attr}="{rewritten}"'

    text = REPO_DOC_LINK_RE.sub(replace_markdown, text)
    text = HTML_ATTR_LINK_RE.sub(replace_html, text)
    return text


def _copy_docs_tree(dest_root: Path) -> None:
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    for name in ["README.md", "project_overview.md", "_static", "_templates"]:
        src = DOCS_ROOT / name
        dst = dest_root / name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    for folder in SITE_COPY_FOLDERS:
        shutil.copytree(DOCS_ROOT / folder, dest_root / folder)

    examples_root = dest_root / "examples"
    examples_root.mkdir(parents=True, exist_ok=True)
    src_examples = REPO_ROOT / "examples"
    notebooks = sorted(src_examples.glob("*.ipynb"))
    for notebook in notebooks:
        shutil.copy2(notebook, examples_root / notebook.name)
    readme_lines = [
        "Last updated: 2026-03-29T00:00:00Z",
        "",
        "# Examples",
        "",
        "This subtree mirrors the canonical checked-in example notebooks for the docs publication build.",
        "The source notebooks remain under the repo-root `examples/` folder.",
        "",
        "Current notebooks:",
    ]
    for notebook in notebooks:
        readme_lines.append(f"- [{notebook.name}](/examples/{notebook.name})")
    readme_lines.extend(
        [
            "",
            "```{toctree}",
            ":maxdepth: 1",
            ":hidden:",
            "",
        ]
    )
    readme_lines.extend(notebook.stem for notebook in notebooks)
    readme_lines.extend(
        [
            "```",
            "",
            "This README is generated as part of the docs publication preparation step.",
        ]
    )
    (examples_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


def _rewrite_docs_tree(dest_root: Path) -> None:
    github_base = _github_base_url()
    github_ref = _github_ref()
    for path in dest_root.rglob("*"):
        if path.suffix.lower() not in MARKDOWN_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        rewritten = _rewrite_markdown_links(text, path, github_base, github_ref, dest_root)
        path.write_text(rewritten, encoding="utf-8")


def prepare_docs(dest_root: Path) -> Path:
    _copy_docs_tree(dest_root)
    _rewrite_docs_tree(dest_root)
    return dest_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a temporary Sphinx/MyST docs source tree.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Destination source tree for the prepared Sphinx docs build.",
    )
    args = parser.parse_args()
    prepared = prepare_docs(args.output_root.resolve())
    print(prepared.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
